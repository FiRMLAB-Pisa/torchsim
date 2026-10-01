#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

#include "_ifunc.hpp"
#include "_threads.hpp"

// MSVC rejects the GNU spelling of both of these outright, and the kernel is
// built by whichever compiler the platform ships.
#if defined(_MSC_VER)
#define TORCHSIM_ALWAYS_INLINE __forceinline
#define TORCHSIM_RESTRICT __restrict
#else
#define TORCHSIM_ALWAYS_INLINE __attribute__((always_inline)) inline
#define TORCHSIM_RESTRICT __restrict__
#endif

namespace {

constexpr std::uint8_t PRE_SHIFT = 1;
constexpr std::uint8_t POST_SHIFT = 2;
constexpr std::uint8_t INVERSION = 4;
constexpr std::uint8_t SPOIL_AFTER = 8;
constexpr std::uint8_t SHIFT_AFTER = 16;
constexpr std::uint8_t RECORD = 32;
constexpr float PI = 3.14159265358979323846F;
// The cubic that gives the three-pool roots is solved in double, where the
// single-precision constant would cost more than the arithmetic around it.
constexpr double TURN_THIRD = 2.09439510239319549231;

// The kernels' input list, mirrored by sequence/_parameters.py: the tissue
// properties, then the packed per-event buffers. The pointer arrays below are
// sized from these, so a parameter appended there is appended here too.
constexpr std::size_t TISSUE_COUNT = 17;
constexpr std::size_t EVENT_COUNT = 9;
constexpr std::size_t PACKED_COUNT = TISSUE_COUNT + EVENT_COUNT;
// Every tissue property carries a gradient; of the event buffers only
// duration, flip and phase do.
constexpr std::size_t FLOAT_COUNT = TISSUE_COUNT + 3;

// Where the transmit pair sits among the tissue properties.
constexpr std::size_t B1_INDEX = 3;
constexpr std::size_t B1_PHASE_INDEX = 4;

// Rows of one voxel each that a tissue parameter's gradient takes. A pulse
// reaches only the shim it drives, so the transmit pair takes a row per shim;
// every other property belongs to the voxel alone and takes one.
inline std::size_t tissue_rows(const std::size_t parameter, const std::int64_t shims) {
    return (parameter == B1_INDEX || parameter == B1_PHASE_INDEX)
        ? static_cast<std::size_t>(shims)
        : 1U;
}

// Where each parameter's gradient starts within the plane, in those rows. At a
// single shim every base is its own parameter index, which is the flat layout
// a sequence without a transmit array uses.
struct TissueLayout {
    std::size_t base[TISSUE_COUNT];
    std::size_t rows;

    explicit TissueLayout(const std::int64_t shims) : base{}, rows(0U) {
        for (std::size_t parameter = 0U; parameter < TISSUE_COUNT; ++parameter) {
            base[parameter] = rows;
            rows += tissue_rows(parameter, shims);
        }
    }
};

using torchsim::MIN_WORK_PER_THREAD;
using torchsim::usable_processors;
using torchsim::WorkerPool;
using torchsim::worker_count;

// Which optional terms a launch carries. A property the caller never passed
// is not an input, so its term is out and its gradient is zero -- the same
// reasoning the pool count already runs on. The bit order is the ABI:
// ``feature_mask`` in ``sequence/_parameters.py`` packs it.
enum Feature : int {
    FEATURE_OFF_AXIS = 1 << 0,
    FEATURE_MOVING = 1 << 1,
    FEATURE_DIFFUSING = 1 << 2,
    FEATURE_TRANSMIT = 1 << 3,
    FEATURE_DENSITY = 1 << 4,
    FEATURE_INVERTING = 1 << 5,
};

// What a caller who declares nothing gets, which is every term.
constexpr int FEATURE_ALL = FEATURE_OFF_AXIS | FEATURE_MOVING
    | FEATURE_DIFFUSING | FEATURE_TRANSMIT | FEATURE_DENSITY
    | FEATURE_INVERTING;

struct Buffers {
    const float* t1;
    const float* t2;
    const float* m0;
    const float* b1;
    const float* b1_phase;
    const float* b0;
    const float* inversion_efficiency;
    const float* diffusion;
    const float* velocity;
    // The semisolid pool: the share of the magnetization it holds, the rate it
    // exchanges with the free water at, and its own T1. Its fraction is the
    // gate -- at zero the exchange is diagonal and the pool starts empty.
    const float* bound_fraction;
    const float* bound_exchange;
    const float* t1_bound;
    // The chemically exchanging pool. It carries transverse magnetization, so
    // it has a T2 the semisolid pool does not and an offset from the free
    // water for that magnetization to turn through. Its fraction gates it on
    // the same terms, and the two second pools are alternatives: the dispatch
    // refuses a tissue declaring both rather than carrying three.
    const float* pool_b_fraction;
    const float* pool_b_exchange;
    const float* t1_pool_b;
    const float* t2_pool_b;
    const float* pool_b_shift;
    const float* duration;
    const std::int32_t* kind;
    const float* flip;
    const float* phase;
    const std::uint8_t* action;
    const std::int32_t* output_index;
    // Which row of the transmit buffers each pulse drives. ``b1`` and
    // ``b1_phase`` hold ``shim_count`` rows of ``atom_count``, so a pulse reads
    // its own shim's field rather than one shared by the whole sequence.
    const std::int32_t* shim_index;
    // What a pulse deposits in the bound pool per unit of flip angle squared,
    // and the frequency it is played at. The saturation a voxel takes is
    // ``saturation[event] * theta^2 * G(rf_frequency[event] - b0)``, with
    // ``theta`` the flip the voxel's transmit field gives it -- uncorrected by
    // the slice profile, the bound pool absorbing the power the pulse deposits
    // rather than the rotation the slice-select gradient shapes out of it.
    const float* saturation;
    const float* rf_frequency;
    float* output_real;
    float* output_imag;
    // ``duration``, ``flip`` and ``phase`` are (train_count, event_count)
    // row-major; every other event buffer describes structure shared by all
    // trains. Work items enumerate the (train, atom) product train-major, so a
    // range of consecutive items aligned to ``atom_count`` owns whole trains --
    // which is what lets threads write event gradients without sharing a slot.
    std::int64_t atom_count;
    std::int64_t train_count;
    // The sequence's gradient geometry, which turns a spin velocity into the
    // two rates it drives: ``flow_scale`` in rad/m is the winding an unbalanced
    // gradient puts across a metre, and ``washout_scale`` in 1/m is the
    // reciprocal voxel size. They are separate because washout depends on the
    // speed a spin leaves the voxel at whether or not a gradient is playing,
    // while flow dephasing depends on the winding it crosses.
    float flow_scale;
    float washout_scale;
    // The terms this launch carries, hoisted out of the mask once so a kernel
    // reads three loop-invariant bools rather than masking per state.
    bool off_axis;
    bool moving;
    bool diffusing;
    // The three per-voxel scalars whose identity is one. Absent, the buffer is
    // not read and the multiply is not made.
    bool transmit;
    bool density;
    bool inverting;
    std::int64_t shim_count;
    // The rotation a shaped pulse performs, tabulated over slice position and
    // effective flip angle: rows of ``profile_bins`` knots, eight floats each
    // -- the Cayley-Klein pair and its slope in the flip angle, interleaved so
    // the two knots a read needs are contiguous. Null when the sequence has no
    // table, which is what selects the flip-and-phase operator instead.
    //
    // One pulse shape occupies ``locations`` consecutive rows, and a sequence
    // may play several: ``profile_index`` says which shape an event drives, so
    // a pulse's row is that shape's block plus the voxel's position. Voxels run
    // voxel-major over the slice, so the position is the index modulo
    // ``locations``.
    const float* profile;
    const std::int32_t* profile_index;
    std::int64_t profile_bins;
    // A rotation per pulse per voxel, ``(rows, atom_count, 4)``: the
    // Cayley-Klein pair, real before imaginary. ``dynamic_index`` says which
    // row an event reads. Null unless the sequence drives a pulse whose
    // channel weights vary while it plays.
    const float* dynamic;
    // ``dynamic_index`` runs per train and per event, as the flip and the
    // phase do, because a pulse's rotation belongs to the train that drives
    // it -- flips vary from train to train. That is also what makes a work
    // item's rows its own: work runs (train, atom), so no two of them reach
    // the same entry and the reverse pass needs no accumulation.
    const std::int32_t* dynamic_index;
    std::int64_t locations;
    float profile_step;
    // How well the bound pool absorbs an off-resonance pulse, tabulated over
    // the offset: ``lineshape_bins`` knots of value and slope, evenly spaced by
    // ``lineshape_step`` Hz. The lineshape is even, so the table covers the
    // magnitude of the offset alone. Null when the sequence has no bound pool,
    // which is what selects the single-pool kernel.
    const float* lineshape;
    std::int64_t lineshape_bins;
    float lineshape_step;
    // The exchanging pools of a tabulated run (``sequence/_pools.py``): per
    // atom the equilibrium of the longitudinal pools, then per distinct
    // interval length the longitudinal operator, that operator applied to the
    // equilibrium, and the transverse operator as real and imaginary pairs.
    // ``pool_index`` names the row each (train, event) reads, and runs per
    // train as the durations do. Null unless the run is tabulated.
    const float* pool_values;
    const std::int32_t* pool_index;
    std::int64_t pool_count;
    std::int64_t pool_rows;
    std::int64_t pool_blocks;
    bool pool_semisolid;
};

// Where a tabulated run's entries sit in one atom's table.
struct PoolShape {
    std::size_t longitudinal;
    std::size_t transverse;
    std::size_t rows;
    std::size_t blocks;
    std::size_t row_width;
    std::size_t width;

    explicit PoolShape(const Buffers& buffers)
        : longitudinal(static_cast<std::size_t>(
              1 + buffers.pool_count + (buffers.pool_semisolid ? 1 : 0)
          )),
          transverse(static_cast<std::size_t>(1 + buffers.pool_count)),
          rows(static_cast<std::size_t>(buffers.pool_rows)),
          blocks(static_cast<std::size_t>(buffers.pool_blocks)),
          row_width(longitudinal * longitudinal + longitudinal
                    + 2U * transverse * transverse),
          width(longitudinal + rows * row_width * blocks) {}

    // ``block`` 0 is the operators, 1 their slope along the interval and 2
    // their curvature.
    std::size_t row(const std::int64_t index, const std::size_t block) const {
        return longitudinal
            + (block * rows + static_cast<std::size_t>(index)) * row_width;
    }

    std::size_t restored() const { return longitudinal * longitudinal; }

    std::size_t across() const { return longitudinal * longitudinal + longitudinal; }
};

// Floats one knot of the transition table holds.
constexpr std::int64_t PROFILE_STRIDE = 8;

// Floats one knot of the lineshape table holds: the value and its slope.
constexpr std::int64_t LINESHAPE_STRIDE = 2;

// Per-work-item view of the buffers that vary along the train axis.
struct TrainView {
    const float* duration;
    const float* flip;
    const float* phase;
    std::int64_t atom;
    std::int64_t train;
    std::int64_t output_base;
    std::int64_t event_base;
};

inline TrainView train_view(
    const Buffers& buffers,
    const std::int64_t work,
    const std::int64_t event_count,
    const std::int64_t output_count
) {
    const std::int64_t train = work / buffers.atom_count;
    const std::int64_t atom = work % buffers.atom_count;
    const std::int64_t event_base = train * event_count;
    return TrainView{
        buffers.duration + event_base,
        buffers.flip + event_base,
        buffers.phase + event_base,
        atom,
        train,
        (train * buffers.atom_count + atom) * output_count,
        event_base,
    };
}

struct JvpBuffers {
    Buffers primal;
    const float* t1;
    const float* t2;
    const float* m0;
    const float* b1;
    const float* b1_phase;
    const float* b0;
    const float* inversion_efficiency;
    const float* diffusion;
    const float* velocity;
    // The bound pool's three properties. A single-pool run has no direction
    // to follow along them, and the kernel it selects does not read them.
    const float* bound_fraction;
    const float* exchange_rate;
    const float* t1_bound;
    // The chemically exchanging pool's five. Read only by the kernels that
    // carry it; a run declaring the semisolid pool instead leaves them alone.
    const float* pool_b_fraction;
    const float* pool_b_exchange;
    const float* t1_pool_b;
    const float* t2_pool_b;
    const float* pool_b_shift;
    const float* duration;
    const float* flip;
    const float* phase;
    // A direction along the per-voxel rotations, laid out as they are. Under
    // any other mode the transmit reaches the kernel as a flip and a phase and
    // this is null; under the dynamic one the array is resolved outside, so a
    // direction along a channel weight or a sensitivity arrives here already
    // carried through the integral.
    const float* dynamic;
    // A direction along a tabulated run's table, laid out as it is.
    const float* pool_values;
};

using Complex = std::complex<float>;

struct DualComplex {
    Complex value;
    Complex tangent;
};

struct DualCoefficient {
    Complex value;
    Complex tangent;
};

struct DualFloat {
    float value;
    float tangent;
};

inline DualFloat operator+(const DualFloat a, const DualFloat b) {
    return {a.value + b.value, a.tangent + b.tangent};
}

inline DualFloat operator-(const DualFloat a, const DualFloat b) {
    return {a.value - b.value, a.tangent - b.tangent};
}

inline DualFloat operator*(const DualFloat a, const DualFloat b) {
    return {a.value * b.value, a.tangent * b.value + a.value * b.tangent};
}

inline DualFloat operator*(const float a, const DualFloat b) {
    return {a * b.value, a * b.tangent};
}

inline DualComplex operator+(const DualComplex a, const DualComplex b) {
    return {a.value + b.value, a.tangent + b.tangent};
}

inline DualComplex operator*(const DualComplex a, const DualComplex b) {
    return {a.value * b.value, a.tangent * b.value + a.value * b.tangent};
}

inline DualComplex operator*(const DualFloat a, const DualComplex b) {
    return {a.value * b.value, a.tangent * b.value + a.value * b.tangent};
}

inline DualComplex operator*(const DualComplex a, const DualFloat b) {
    return {a.value * b.value, a.tangent * b.value + a.value * b.tangent};
}

inline DualComplex operator*(const Complex a, const DualComplex b) {
    return {a * b.value, a * b.tangent};
}

inline DualComplex operator-(const DualComplex a, const DualComplex b) {
    return {a.value - b.value, a.tangent - b.tangent};
}

inline DualComplex conjugate(const DualComplex a) {
    return {std::conj(a.value), std::conj(a.tangent)};
}

inline DualFloat real_part(const DualComplex a) {
    return {a.value.real(), a.tangent.real()};
}

inline DualFloat imag_part(const DualComplex a) {
    return {a.value.imag(), a.tangent.imag()};
}

inline DualComplex to_complex(const DualFloat a) {
    return {Complex(a.value, 0.0F), Complex(a.tangent, 0.0F)};
}

inline DualFloat dual_exp(const DualFloat a) {
    const float value = std::exp(a.value);
    return {value, value * a.tangent};
}

inline DualFloat dual_cos(const DualFloat a) {
    return {std::cos(a.value), -std::sin(a.value) * a.tangent};
}

inline DualFloat dual_sin(const DualFloat a) {
    return {std::sin(a.value), std::cos(a.value) * a.tangent};
}

// exp(i * angle) for a dual angle
inline DualComplex dual_polar(const DualFloat angle) {
    const Complex value = std::polar(1.0F, angle.value);
    return {value, Complex(0.0F, angle.tangent) * value};
}

// The value a scalar carries, whether or not it carries a tangent beside it.
// Lets a shared implementation branch on the primal without knowing its mode.
inline float primal(const float a) {
    return a;
}

inline float primal(const DualFloat a) {
    return a.value;
}

// Clamped at zero, because the only square root the state machine takes is of
// a discriminant that is non-negative by construction and can still land a
// rounding below it. The tangent divides by the root, so a caller must keep
// the origin -- where the root has no derivative -- on its series branch.
inline float root(const float a) {
    return std::sqrt(std::max(a, 0.0F));
}

inline DualFloat root(const DualFloat a) {
    const float value = std::sqrt(std::max(a.value, 0.0F));
    return {value, value > 0.0F ? 0.5F * a.tangent / value : 0.0F};
}

inline DualFloat dual_reciprocal(const DualFloat a) {
    const float inverse = 1.0F / a.value;
    return {inverse, -a.tangent * inverse * inverse};
}

// ---------------------------------------------------------------------------
// Diffusion damping.
//
// Over a free-precession interval a state at dephasing order l accumulates a
// b-factor weighted by l^2 while longitudinal and by l^2 + l + 1/3 while
// transverse. The rate arrives already carrying the sequence's gradient
// geometry, so ``rate * dt`` is the b-factor an order-1 state accumulates.
//
// Order zero has longitudinal weight zero, so its damping is exactly one: the
// affine recovery term needs no special case.
// ---------------------------------------------------------------------------

inline float longitudinal_weight(const std::size_t state) {
    const float order = static_cast<float>(state);
    return order * order;
}

inline float transverse_weight(const std::size_t state) {
    const float order = static_cast<float>(state);
    return order * order + order + (1.0F / 3.0F);
}

inline float exponential(const float a) {
    return std::exp(a);
}

inline DualFloat exponential(const DualFloat a) {
    return dual_exp(a);
}

inline float reciprocal(const float a) {
    return 1.0F / a;
}

inline DualFloat reciprocal(const DualFloat a) {
    return dual_reciprocal(a);
}

// The same handful of operations in double, for the one operator the state
// machine forms there. A 2x2's closed form loses accuracy like the interval; a
// 3x3's loses it like the square of it, because the answer's entries are order
// one while the terms that build them are order |L dt|^2. See
// tests/epg/test_three_pool.py, which measures both.
inline double primal(const double a) {
    return a;
}

inline double root(const double a) {
    return std::sqrt(std::max(a, 0.0));
}

inline double exponential(const double a) {
    return std::exp(a);
}

inline double reciprocal(const double a) {
    return 1.0 / a;
}

inline double arc_cosine(const double a) {
    return std::acos(a);
}

inline double cosine(const double a) {
    return std::cos(a);
}

inline double sine(const double a) {
    return std::sin(a);
}

// A direction carried alongside a double, for the three-pool step's forward
// mode. The two-pool steps run in single and carry ``DualFloat``; only this
// one operator needs the wider pair.
struct DualDouble {
    double value;
    double tangent;
};

inline DualDouble operator+(const DualDouble a, const DualDouble b) {
    return {a.value + b.value, a.tangent + b.tangent};
}

inline DualDouble operator-(const DualDouble a, const DualDouble b) {
    return {a.value - b.value, a.tangent - b.tangent};
}

inline DualDouble operator*(const DualDouble a, const DualDouble b) {
    return {a.value * b.value, a.tangent * b.value + a.value * b.tangent};
}

inline double primal(const DualDouble a) {
    return a.value;
}

inline DualDouble root(const DualDouble a) {
    const double value = std::sqrt(std::max(a.value, 0.0));
    return {value, value > 0.0 ? 0.5 * a.tangent / value : 0.0};
}

inline DualDouble exponential(const DualDouble a) {
    const double value = std::exp(a.value);
    return {value, value * a.tangent};
}

inline DualDouble reciprocal(const DualDouble a) {
    const double inverse = 1.0 / a.value;
    return {inverse, -a.tangent * inverse * inverse};
}

inline DualDouble arc_cosine(const DualDouble a) {
    const double slope = -1.0 / std::sqrt(std::max(1.0 - a.value * a.value, 1e-300));
    return {std::acos(a.value), slope * a.tangent};
}

inline DualDouble cosine(const DualDouble a) {
    return {std::cos(a.value), -std::sin(a.value) * a.tangent};
}

inline DualDouble sine(const DualDouble a) {
    return {std::sin(a.value), std::cos(a.value) * a.tangent};
}

// The three-pool step is the only place the state machine changes width, so
// the narrowing sits beside the widening rather than at either call site.
inline DualDouble widen_dual(const DualFloat a) {
    return {static_cast<double>(a.value), static_cast<double>(a.tangent)};
}

inline DualFloat narrow_dual(const DualDouble a) {
    return {static_cast<float>(a.value), static_cast<float>(a.tangent)};
}

inline bool is_zero(const float a) {
    return a == 0.0F;
}

inline bool is_zero(const DualFloat a) {
    return a.value == 0.0F && a.tangent == 0.0F;
}

// The same three operations on a complex scalar, for the transverse pair,
// whose generator carries the chemical shift on its diagonal and so is complex
// wherever the pools sit at different offsets.
//
// ``root`` takes whichever branch the library gives it. The operator it feeds
// is even in the root -- ``cosh`` and ``sinh(d)/d`` both are -- so it is a
// function of the discriminant alone and the branch cut is unreachable.
inline Complex exponential(const Complex a) {
    return std::exp(a);
}

inline Complex root(const Complex a) {
    return std::sqrt(a);
}

inline Complex reciprocal(const Complex a) {
    return 1.0F / a;
}

// How far a complex scalar is from the origin, squared, for the branch that
// decides whether the series takes over from the division.
inline float primal_norm(const Complex a) {
    return std::norm(a);
}

// Promoting a real scalar to the complex one that goes with it, along the real
// axis and along the imaginary one.
inline Complex widen(const float a) {
    return Complex(a, 0.0F);
}

inline Complex as_imaginary(const float a) {
    return Complex(0.0F, a);
}

inline float real_part(const Complex a) {
    return a.real();
}

inline DualComplex widen(const DualFloat a) {
    return to_complex(a);
}

inline DualComplex as_imaginary(const DualFloat a) {
    return {Complex(0.0F, a.value), Complex(0.0F, a.tangent)};
}

// The same three operations carried along a direction. A complex square root
// divides its tangent by twice itself, so the caller keeps the origin -- where
// it has no derivative -- on the series branch, exactly as the real one does.
inline DualComplex exponential(const DualComplex a) {
    const Complex value = std::exp(a.value);
    return {value, value * a.tangent};
}

inline DualComplex root(const DualComplex a) {
    const Complex value = std::sqrt(a.value);
    if (value == Complex{}) {
        return {value, Complex{}};
    }
    return {value, a.tangent / (2.0F * value)};
}

inline DualComplex reciprocal(const DualComplex a) {
    const Complex inverse = 1.0F / a.value;
    return {inverse, Complex{} - a.tangent * inverse * inverse};
}

inline float primal_norm(const DualComplex a) {
    return std::norm(a.value);
}

inline void set_unit(float& target) {
    target = 1.0F;
}

inline void set_unit(DualFloat& target) {
    target = DualFloat{1.0F, 0.0F};
}

// Per-order damping factors for one interval, held across events so a train
// with no diffusion never evaluates a transcendental.
template <typename Scalar>
struct Damping {
    std::vector<Scalar> longitudinal;
    std::vector<Scalar> transverse;
    bool damped;

    explicit Damping(const std::size_t states)
        : longitudinal(states), transverse(states), damped(true) {
        reset();
    }

    void reset() {
        if (!damped) {
            return;
        }
        for (std::size_t state = 0; state < longitudinal.size(); ++state) {
            set_unit(longitudinal[state]);
            set_unit(transverse[state]);
        }
        damped = false;
    }

    // Set the factors for one interval, leaving them at one where there is no
    // diffusion to apply.
    void set(const Scalar rate, const Scalar dt) {
        if (is_zero(rate)) {
            reset();
            return;
        }
        const Scalar b_factor = rate * dt;
        for (std::size_t state = 0; state < longitudinal.size(); ++state) {
            longitudinal[state] =
                exponential(-longitudinal_weight(state) * b_factor);
            transverse[state] =
                exponential(-transverse_weight(state) * b_factor);
        }
        damped = true;
    }
};

// Phase each dephasing order turns through over one interval. The rate is the
// winding per unit order per second, so a longitudinal state at order l turns
// through l * rate * dt; the transverse states sit half an order further along
// the gradient. Order zero is left alone while longitudinal, so the recovery
// term is unaffected.
inline void flow_turn(
    const float rate,
    const float dt,
    const std::size_t state,
    float& longitudinal,
    float& transverse
) {
    const float turn = rate * dt;
    const float order = static_cast<float>(state);
    longitudinal = -order * turn;
    transverse = -(order + 0.5F) * turn;
}

// The same turn carried on dual numbers, for the kernels that differentiate.
template <typename Scalar>
inline void flow_turn_dual(
    const Scalar rate,
    const Scalar dt,
    const std::size_t state,
    Scalar& longitudinal,
    Scalar& transverse
) {
    const Scalar turn = rate * dt;
    const float order = static_cast<float>(state);
    longitudinal = (-order) * turn;
    transverse = (-(order + 0.5F)) * turn;
}

// A per-voxel scalar whose identity is one: absent, the buffer is not read and
// the multiply that would have used it is not made.
inline float scaled(const float value, const float factor, const bool live) {
    return live ? value * factor : value;
}

// The same, for a kernel carrying a forward direction: an absent scalar sits at
// one and its direction at zero, and neither buffer is read.
// A rate whose identity is zero, for a kernel carrying a forward direction.
inline DualFloat held_rate(
    const float* const value,
    const float* const direction,
    const std::int64_t atom,
    const bool live
) {
    return live ? DualFloat{value[atom], direction[atom]} : DualFloat{0.0F, 0.0F};
}

inline DualFloat held(
    const float* const value,
    const float* const direction,
    const std::int64_t atom,
    const bool live
) {
    return live ? DualFloat{value[atom], direction[atom]} : DualFloat{1.0F, 0.0F};
}

// A turn through an angle, or the plain scaling a zero angle amounts to.
// Written out rather than left to ``std::polar``, which takes a sine and a
// cosine whatever it is given: a sequence with nothing to turn the states
// through would otherwise pay two transcendentals per state per event for a
// factor of one.
inline Complex turned(
    const float magnitude, const float angle, const bool turning
) {
    return turning ? std::polar(magnitude, angle) : Complex{magnitude, 0.0F};
}

// The fraction of a voxel's spins replaced by inflowing ones over an interval,
// at a rate of ``|v| / voxel_size`` per second. Clamped at one: a spin cannot
// leave the voxel more than once, and past that the whole voxel has turned over.
//
// The inflowing spins are taken to be fully relaxed and unexcited -- no
// transverse magnetization and unit longitudinal -- which is the single
// compartment a state machine over one voxel can see. That makes washout an
// affine map of exactly the shape longitudinal recovery already has, and
//
//     wout * (Z * e1 + (1 - e1)) + win  ==  Z * (e1 * wout) + (1 - e1 * wout)
//
// so it needs no term of its own: scaling e1 and e2 by ``wout`` carries it,
// derivatives included.
inline float washout_out(const float rate, const float dt) {
    return 1.0F - std::min(rate * dt, 1.0F);
}

// The same fraction carried on dual numbers. Past the clamp the interval has
// replaced the voxel outright and nothing further depends on the rate, so the
// tangent is zero there.
inline DualFloat washout_out(const DualFloat rate, const DualFloat dt) {
    const DualFloat fraction = rate * dt;
    if (fraction.value >= 1.0F) {
        return DualFloat{0.0F, 0.0F};
    }
    return DualFloat{1.0F - fraction.value, -fraction.tangent};
}

// Which way a velocity points, and nothing at all where it is zero: washout
// depends on the speed, and |v| has no derivative at the origin.
inline float speed_direction(const float velocity) {
    return static_cast<float>(velocity > 0.0F) - static_cast<float>(velocity < 0.0F);
}

// The longitudinal step of a two-pool system: ``Z <- E1 Z`` over one interval,
// with the recovery it adds to the zeroth dephasing order.
//
// ``E1 = expm(L t)`` for the 2x2 generator ``L = K - diag(R1a, R1b)``, which
// has an exact closed form,
//
//     tau = tr(Lt)/2   d^2 = tau^2 - det(Lt)   expm = e^tau [cosh d I + (sinh
//     d / d)(Lt - tau I)]
//
// and whose discriminant ``d^2 = ((L11-L22)t/2)^2 + k_ab k_ba t^2`` is a sum of
// a square and a product of two non-negative rates, so it never leaves the real
// line: no complex branch, and one square root per event rather than an
// eigendecomposition. ``sinh(d)/d`` is taken by series near the origin, where
// the root itself has no derivative.
//
// The equilibrium each pool relaxes toward is its own fraction -- ``L (1-f, f)
// = -C`` holds identically -- so the recovery is ``(I - E1) (1-f, f)`` and
// needs no 2x2 solve.
// Which second pools a kernel carries, if any. Naming the combination rather
// than passing two flags is what keeps the instantiations to one per case
// instead of one per flag product, and lets the blocks below read as the
// physics they are.
//
// Free water always carries ``F+``, ``F-`` and ``Z``. The chemically
// exchanging pool carries all three as well; the semisolid pool carries ``Z``
// alone, so the transverse operator is the same 2x2 whether or not it is
// there and only the longitudinal one grows.
enum class Pools {
    ONE,
    SEMISOLID,
    EXCHANGING,
    THREE,
    // Any number of exchanging pools, with or without a semisolid one, read
    // from operators tabulated per interval. Never a template argument: the
    // tabulated kernels are their own.
    TABULATED,
};

// How a pulse's rotation is reached, for the same reason the pools are one
// enum rather than a product of flags: the three are alternatives, not
// features that combine.
//
// ``INSTANT`` turns through a flip angle and a phase. ``PROFILED`` reads the
// rotation a shaped pulse performs from a table over slice position and
// effective flip. ``DYNAMIC`` reads it per voxel, which is what a pulse whose
// channel weights vary while it plays needs -- and which subsumes a profile,
// since a rotation integrated at the voxel's own position is one.
enum class RfMode {
    INSTANT,
    PROFILED,
    DYNAMIC,
};

template <typename Scalar>
struct TwoPoolStep {
    Scalar e11;
    Scalar e12;
    Scalar e21;
    Scalar e22;
    Scalar recovery_free;
    Scalar recovery_bound;
};

template <typename Scalar>
inline TwoPoolStep<Scalar> two_pool_step(
    const Scalar r1_free,
    const Scalar r1_bound,
    const Scalar exchange,
    const Scalar bound,
    const Scalar dt,
    const Scalar attenuation
) {
    const Scalar free = Scalar{1.0F} - bound;
    const Scalar kab = exchange * bound;
    const Scalar kba = exchange * free;
    const Scalar l11 = (Scalar{} - kab - r1_free) * dt;
    const Scalar l12 = kba * dt;
    const Scalar l21 = kab * dt;
    const Scalar l22 = (Scalar{} - kba - r1_bound) * dt;

    const Scalar half_trace = 0.5F * (l11 + l22);
    const Scalar half_gap = 0.5F * (l11 - l22);
    const Scalar square = half_gap * half_gap + l12 * l21;
    // tau +/- d are the eigenvalues, both non-positive for a decaying system,
    // so their exponentials are bounded by one. Formed that way rather than as
    // e^tau cosh(d), which over a long interval is an underflow times an
    // overflow.
    const Scalar delta = root(square);
    const Scalar upper = exponential(half_trace + delta);
    const Scalar lower = exponential(half_trace - delta);
    const Scalar cosine = 0.5F * (upper + lower);
    // sinh(d)/d by series where the root has no derivative of its own. The
    // threshold is on the discriminant, so both the value and the tangent
    // leave the branch before the division does any damage.
    const Scalar scale = primal(square) > 1e-12F
        ? 0.5F * (upper - lower) * reciprocal(delta)
        : exponential(half_trace)
            * (Scalar{1.0F} + (1.0F / 6.0F) * square
               + (1.0F / 120.0F) * (square * square));

    TwoPoolStep<Scalar> step{};
    step.e11 = attenuation * (cosine + scale * half_gap);
    step.e12 = attenuation * scale * l12;
    step.e21 = attenuation * scale * l21;
    step.e22 = attenuation * (cosine - scale * half_gap);
    step.recovery_free = free - (step.e11 * free + step.e12 * bound);
    step.recovery_bound = bound - (step.e21 * free + step.e22 * bound);
    return step;
}

// What a cotangent on the two-pool step leaves on the six numbers it was
// formed from.
template <typename Scalar>
struct TwoPoolGradient {
    Scalar r1_free;
    Scalar r1_bound;
    Scalar exchange;
    Scalar bound;
    Scalar dt;
    Scalar attenuation;
};

// The reverse sweep of ``two_pool_step``, which recomputes the forward rather
// than carrying it across the event: the whole thing is a handful of
// transcendentals once per interval, against a state loop that runs per
// dephasing order.
//
// Where the discriminant is small the value is still formed from the two
// eigenvalues -- a sum, which loses nothing -- but the derivative is taken from
// the series, because ``d cosh(d)/d(d^2)`` reached through ``(e^{t+d} -
// e^{t-d})/2d`` is a cancellation divided by a small number.
template <typename Scalar>
inline TwoPoolGradient<Scalar> two_pool_step_adjoint(
    const Scalar r1_free,
    const Scalar r1_bound,
    const Scalar exchange,
    const Scalar bound,
    const Scalar dt,
    const Scalar attenuation,
    const Scalar bar_e11,
    const Scalar bar_e12,
    const Scalar bar_e21,
    const Scalar bar_e22,
    const Scalar bar_recovery_free,
    const Scalar bar_recovery_bound
) {
    const Scalar free = Scalar{1.0F} - bound;
    const Scalar kab = exchange * bound;
    const Scalar kba = exchange * free;
    const Scalar l11 = (Scalar{} - kab - r1_free) * dt;
    const Scalar l12 = kba * dt;
    const Scalar l21 = kab * dt;
    const Scalar l22 = (Scalar{} - kba - r1_bound) * dt;

    const Scalar half_trace = 0.5F * (l11 + l22);
    const Scalar half_gap = 0.5F * (l11 - l22);
    const Scalar square = half_gap * half_gap + l12 * l21;
    const bool series = !(primal(square) > 1e-12F);
    const Scalar delta = root(square);
    const Scalar upper = exponential(half_trace + delta);
    const Scalar lower = exponential(half_trace - delta);
    const Scalar plain = exponential(half_trace);
    const Scalar cosine = 0.5F * (upper + lower);
    const Scalar scale = series
        ? plain
            * (Scalar{1.0F} + (1.0F / 6.0F) * square
               + (1.0F / 120.0F) * (square * square))
        : 0.5F * (upper - lower) * reciprocal(delta);

    const Scalar bare_11 = cosine + scale * half_gap;
    const Scalar bare_12 = scale * l12;
    const Scalar bare_21 = scale * l21;
    const Scalar bare_22 = cosine - scale * half_gap;

    // The recovery reaches the operator's four entries and the two fractions.
    const Scalar carried_11 = bar_e11 - bar_recovery_free * free;
    const Scalar carried_12 = bar_e12 - bar_recovery_free * bound;
    const Scalar carried_21 = bar_e21 - bar_recovery_bound * free;
    const Scalar carried_22 = bar_e22 - bar_recovery_bound * bound;
    Scalar bar_free = bar_recovery_free * (Scalar{1.0F} - attenuation * bare_11)
        - bar_recovery_bound * (attenuation * bare_21);
    Scalar bar_bound = bar_recovery_bound * (Scalar{1.0F} - attenuation * bare_22)
        - bar_recovery_free * (attenuation * bare_12);

    const Scalar bar_attenuation = carried_11 * bare_11 + carried_12 * bare_12
        + carried_21 * bare_21 + carried_22 * bare_22;
    const Scalar scaled_11 = attenuation * carried_11;
    const Scalar scaled_12 = attenuation * carried_12;
    const Scalar scaled_21 = attenuation * carried_21;
    const Scalar scaled_22 = attenuation * carried_22;

    const Scalar bar_cosine = scaled_11 + scaled_22;
    const Scalar bar_scale = (scaled_11 - scaled_22) * half_gap
        + scaled_12 * l12 + scaled_21 * l21;
    Scalar bar_half_gap = scale * (scaled_11 - scaled_22);
    Scalar bar_l12 = scale * scaled_12;
    Scalar bar_l21 = scale * scaled_21;

    Scalar bar_half_trace{};
    Scalar bar_square{};
    if (series) {
        bar_half_trace = bar_cosine * cosine + bar_scale * scale;
        bar_square = plain
            * (bar_cosine * (Scalar{0.5F} + (1.0F / 12.0F) * square)
               + bar_scale * (Scalar{1.0F / 6.0F} + (1.0F / 60.0F) * square));
    } else {
        const Scalar inverse = reciprocal(delta);
        const Scalar bar_upper = 0.5F * (bar_cosine + bar_scale * inverse);
        const Scalar bar_lower = 0.5F * (bar_cosine - bar_scale * inverse);
        bar_half_trace = bar_upper * upper + bar_lower * lower;
        const Scalar bar_delta = bar_upper * upper - bar_lower * lower
            - bar_scale * scale * inverse;
        bar_square = 0.5F * bar_delta * inverse;
    }

    bar_half_gap = bar_half_gap + 2.0F * bar_square * half_gap;
    bar_l12 = bar_l12 + bar_square * l21;
    bar_l21 = bar_l21 + bar_square * l12;

    const Scalar bar_l11 = 0.5F * (bar_half_trace + bar_half_gap);
    const Scalar bar_l22 = 0.5F * (bar_half_trace - bar_half_gap);

    const Scalar bar_kab = (bar_l21 - bar_l11) * dt;
    const Scalar bar_kba = (bar_l12 - bar_l22) * dt;
    const Scalar bar_dt = bar_l11 * (Scalar{} - kab - r1_free)
        + bar_l12 * kba + bar_l21 * kab
        + bar_l22 * (Scalar{} - kba - r1_bound);

    bar_bound = bar_bound + bar_kab * exchange;
    bar_free = bar_free + bar_kba * exchange;

    TwoPoolGradient<Scalar> gradient{};
    gradient.r1_free = Scalar{} - bar_l11 * dt;
    gradient.r1_bound = Scalar{} - bar_l22 * dt;
    gradient.exchange = bar_kab * bound + bar_kba * free;
    gradient.bound = bar_bound - bar_free;
    gradient.dt = bar_dt;
    gradient.attenuation = bar_attenuation;
    return gradient;
}

// The three-pool longitudinal step: free water beside both second pools at
// once, over one interval.
//
// This is the one operator the state machine forms in double, and the reason
// is measured in tests/epg/test_three_pool.py: a 2x2's closed form loses
// accuracy like the interval, a 3x3's like the square of it, because the
// answer's entries are order one while the terms that build them are order
// |L dt|^2. That is intrinsic to writing the answer as a polynomial in the
// generator, so it is met with precision rather than with rearrangement. The
// operator is formed once per interval and narrowed as soon as it is an
// operator, so the cost stays out of the per-order loop.
//
// Two branches, by how far apart the eigenvalues are. Where they are close the
// exponential's own series is reduced modulo the characteristic polynomial,
// which forms no root at all -- so it stays differentiable exactly where the
// roots stop being. Where they are far apart the interpolating polynomial is
// taken in Newton form at the three roots, each of which is non-positive, so
// each exponential of one is at most one and a long interval cannot overflow.
//
// Free water is pool a, the chemically exchanging pool b and the semisolid
// pool c. Each second pool exchanges with the free water and not with the
// other, which is what makes each two-pool system a limit of this one.
constexpr double SPREAD_CUT = 1.0;
constexpr double SINCH_CUT = 1e-4;
// Where two roots meet the sorted roots have a vertical tangent the operator
// itself does not, so the arc cosine is held a hair off its endpoints. At this
// guard the operator is bit for bit what an unguarded one returns and the bias
// at an exact degeneracy is four orders under the float32 that follows.
constexpr double ARG_GUARD = 1e-16;
constexpr int SERIES_TERMS = 16;

template <typename Scalar>
struct ThreePoolStep {
    Scalar entry[3][3];
    Scalar recovery[3];
};

template <typename Scalar>
inline Scalar three_pool_minors(const Scalar (&a)[3][3]) {
    return a[0][0] * a[1][1] - a[0][1] * a[1][0]
        + a[0][0] * a[2][2] - a[0][2] * a[2][0]
        + a[1][1] * a[2][2] - a[1][2] * a[2][1];
}

template <typename Scalar>
inline Scalar three_pool_determinant(const Scalar (&a)[3][3]) {
    return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
}

// ``[a, b] exp``, by series where the two points nearly meet. sinh(d)/d is
// even in the gap, so the series is a polynomial in its square.
template <typename Scalar>
inline Scalar three_pool_difference(const Scalar lower, const Scalar upper) {
    const Scalar half = Scalar{0.5} * (upper - lower);
    if (std::fabs(primal(half)) < SINCH_CUT) {
        const Scalar square = half * half;
        const Scalar series = Scalar{1.0}
            + square * (Scalar{1.0 / 6.0}
                + square * (Scalar{1.0 / 120.0} + square * Scalar{1.0 / 5040.0}));
        return exponential(Scalar{0.5} * (lower + upper)) * series;
    }
    return (exponential(upper) - exponential(lower)) * reciprocal(upper - lower);
}

template <typename Scalar>
inline ThreePoolStep<Scalar> three_pool_step(
    const Scalar r1_free,
    const Scalar r1_pool_b,
    const Scalar r1_bound,
    const Scalar exchange_b,
    const Scalar exchange_c,
    const Scalar fraction_b,
    const Scalar fraction_c,
    const Scalar dt,
    const Scalar attenuation
) {
    const Scalar free = Scalar{1.0} - fraction_b - fraction_c;
    const Scalar kab = exchange_b * fraction_b;
    const Scalar kba = exchange_b * free;
    const Scalar kac = exchange_c * fraction_c;
    const Scalar kca = exchange_c * free;
    Scalar a[3][3]{};
    a[0][0] = (Scalar{} - kab - kac - r1_free) * dt;
    a[0][1] = kba * dt;
    a[0][2] = kca * dt;
    a[1][0] = kab * dt;
    a[1][1] = (Scalar{} - kba - r1_pool_b) * dt;
    a[2][0] = kac * dt;
    a[2][2] = (Scalar{} - kca - r1_bound) * dt;

    const Scalar third = Scalar{1.0 / 3.0} * (a[0][0] + a[1][1] + a[2][2]);
    Scalar shifted[3][3];
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            shifted[row][column] =
                row == column ? a[row][column] - third : a[row][column];
        }
    }
    const Scalar minors = three_pool_minors(shifted);
    const Scalar determinant = three_pool_determinant(shifted);

    Scalar operator_[3][3]{};
    // The shifted generator's roots sum to zero, so the sum of their squares
    // is -2 * minors and none is larger than the root of that.
    if (-2.0 * primal(minors) < SPREAD_CUT * SPREAD_CUT) {
        // x^k reduced modulo x^3 + minors x - determinant, carried as the
        // three coefficients of 1, x and x^2 and stepped by the recurrence
        // that reduction gives.
        Scalar flat{1.0};
        Scalar linear{};
        Scalar square{};
        Scalar sum_flat{1.0};
        Scalar sum_linear{};
        Scalar sum_square{};
        double factorial = 1.0;
        for (int order = 1; order < SERIES_TERMS; ++order) {
            const Scalar next_flat = square * determinant;
            const Scalar next_linear = flat - square * minors;
            const Scalar next_square = linear;
            flat = next_flat;
            linear = next_linear;
            square = next_square;
            factorial *= static_cast<double>(order);
            const Scalar weight{1.0 / factorial};
            sum_flat = sum_flat + weight * flat;
            sum_linear = sum_linear + weight * linear;
            sum_square = sum_square + weight * square;
        }
        const Scalar scale = exponential(third);
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                Scalar squared{};
                for (int inner = 0; inner < 3; ++inner) {
                    squared = squared + shifted[row][inner] * shifted[inner][column];
                }
                const Scalar identity = row == column ? sum_flat : Scalar{};
                operator_[row][column] = scale
                    * (identity + sum_linear * shifted[row][column]
                       + sum_square * squared);
            }
        }
    } else {
        const Scalar radius = root(Scalar{-1.0 / 3.0} * minors);
        // The depressed cubic's constant term is minus this determinant, and
        // the trigonometric solution wants that term over twice the radius
        // cubed -- so the two signs cancel and this one is positive.
        Scalar argument =
            Scalar{0.5} * determinant * reciprocal(radius * radius * radius);
        constexpr double limit = 1.0 - ARG_GUARD;
        if (primal(argument) > limit) {
            argument = Scalar{limit};
        } else if (primal(argument) < -limit) {
            argument = Scalar{-limit};
        }
        const Scalar angle = Scalar{1.0 / 3.0} * arc_cosine(argument);
        Scalar roots[3];
        for (int turn = 0; turn < 3; ++turn) {
            roots[turn] = Scalar{2.0} * radius
                * cosine(angle - Scalar{TURN_THIRD * turn}) + third;
        }
        for (int pass = 0; pass < 2; ++pass) {
            for (int index = 0; index + 1 < 3; ++index) {
                if (primal(roots[index]) > primal(roots[index + 1])) {
                    const Scalar held = roots[index];
                    roots[index] = roots[index + 1];
                    roots[index + 1] = held;
                }
            }
        }
        const Scalar low = three_pool_difference(roots[0], roots[1]);
        const Scalar high = three_pool_difference(roots[1], roots[2]);
        const Scalar second = (high - low) * reciprocal(roots[2] - roots[0]);
        const Scalar leading = exponential(roots[0]);
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                Scalar product{};
                for (int inner = 0; inner < 3; ++inner) {
                    const Scalar first = row == inner
                        ? a[row][inner] - roots[0] : a[row][inner];
                    const Scalar next = inner == column
                        ? a[inner][column] - roots[1] : a[inner][column];
                    product = product + first * next;
                }
                const Scalar shift = row == column
                    ? a[row][column] - roots[0] : a[row][column];
                const Scalar identity = row == column ? leading : Scalar{};
                operator_[row][column] = identity + low * shift + second * product;
            }
        }
    }

    ThreePoolStep<Scalar> step{};
    const Scalar equilibrium[3] = {free, fraction_b, fraction_c};
    for (int row = 0; row < 3; ++row) {
        Scalar carried{};
        for (int column = 0; column < 3; ++column) {
            step.entry[row][column] = attenuation * operator_[row][column];
            carried = carried + step.entry[row][column] * equilibrium[column];
        }
        step.recovery[row] = equilibrium[row] - carried;
    }
    return step;
}

// What a cotangent on the three-pool step leaves on the nine numbers it was
// formed from.
template <typename Scalar>
struct ThreePoolGradient {
    Scalar r1_free;
    Scalar r1_pool_b;
    Scalar r1_bound;
    Scalar exchange_b;
    Scalar exchange_c;
    Scalar fraction_b;
    Scalar fraction_c;
    Scalar dt;
    Scalar attenuation;
};

// The reverse sweep of ``three_pool_step``, in the same double the forward
// takes. It recomputes the forward rather than carrying it across the event:
// the whole thing is a handful of transcendentals once per interval, against a
// state loop that runs per dephasing order.
//
// Both branches are swept, each by the algebra its own forward used. The
// series one is a linear recurrence in the two invariants, so its reverse is
// that recurrence run backwards over the sixteen triples the forward left; the
// eigenvalue one walks back through the roots, whose sorting is a permutation
// and so sends each cotangent to the root it came from.
// The reverse of ``three_pool_difference``, accumulating onto both points.
template <typename Scalar>
inline void three_pool_difference_adjoint(
    const Scalar lower,
    const Scalar upper,
    const Scalar seed,
    Scalar& bar_lower,
    Scalar& bar_upper
) {
    const Scalar half = Scalar{0.5} * (upper - lower);
    if (std::fabs(primal(half)) < SINCH_CUT) {
        const Scalar square = half * half;
        const Scalar series = Scalar{1.0}
            + square * (Scalar{1.0 / 6.0}
                + square * (Scalar{1.0 / 120.0} + square * Scalar{1.0 / 5040.0}));
        const Scalar slope = Scalar{1.0 / 6.0}
            + square * (Scalar{1.0 / 60.0} + square * Scalar{1.0 / 1680.0});
        const Scalar lift = exponential(Scalar{0.5} * (lower + upper));
        const Scalar along_mid = Scalar{0.5} * seed * lift * series;
        const Scalar along_gap = seed * lift * slope * half;
        bar_lower = bar_lower + along_mid - along_gap;
        bar_upper = bar_upper + along_mid + along_gap;
        return;
    }
    const Scalar gap = upper - lower;
    const Scalar inverse = reciprocal(gap);
    const Scalar low = exponential(lower);
    const Scalar high = exponential(upper);
    const Scalar value = (high - low) * inverse;
    bar_lower = bar_lower + seed * (value - low) * inverse;
    bar_upper = bar_upper + seed * (high - value) * inverse;
}

template <typename Scalar>
inline ThreePoolGradient<Scalar> three_pool_step_adjoint(
    const Scalar r1_free,
    const Scalar r1_pool_b,
    const Scalar r1_bound,
    const Scalar exchange_b,
    const Scalar exchange_c,
    const Scalar fraction_b,
    const Scalar fraction_c,
    const Scalar dt,
    const Scalar attenuation,
    const Scalar (&bar_entry)[3][3],
    const Scalar (&bar_recovery)[3]
) {
    const Scalar free = Scalar{1.0} - fraction_b - fraction_c;
    const Scalar kab = exchange_b * fraction_b;
    const Scalar kba = exchange_b * free;
    const Scalar kac = exchange_c * fraction_c;
    const Scalar kca = exchange_c * free;
    Scalar a[3][3]{};
    a[0][0] = (Scalar{} - kab - kac - r1_free) * dt;
    a[0][1] = kba * dt;
    a[0][2] = kca * dt;
    a[1][0] = kab * dt;
    a[1][1] = (Scalar{} - kba - r1_pool_b) * dt;
    a[2][0] = kac * dt;
    a[2][2] = (Scalar{} - kca - r1_bound) * dt;

    const Scalar third = Scalar{1.0 / 3.0} * (a[0][0] + a[1][1] + a[2][2]);
    Scalar shifted[3][3];
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            shifted[row][column] =
                row == column ? a[row][column] - third : a[row][column];
        }
    }
    const Scalar minors = three_pool_minors(shifted);
    const Scalar determinant = three_pool_determinant(shifted);
    const bool close = -2.0 * primal(minors) < SPREAD_CUT * SPREAD_CUT;

    // ---- the recovery and the attenuation, which both branches share ----
    const Scalar equilibrium[3] = {free, fraction_b, fraction_c};
    Scalar carried[3][3];
    Scalar bar_operator[3][3];
    Scalar bar_attenuation{};
    Scalar bar_equilibrium[3] = {Scalar{}, Scalar{}, Scalar{}};
    const ThreePoolStep<Scalar> replay = three_pool_step(
        r1_free, r1_pool_b, r1_bound, exchange_b, exchange_c,
        fraction_b, fraction_c, dt, Scalar{1.0}
    );
    for (int row = 0; row < 3; ++row) {
        bar_equilibrium[row] = bar_equilibrium[row] + bar_recovery[row];
        for (int column = 0; column < 3; ++column) {
            // The replay is taken at unit attenuation, so the bare operator
            // the recovery and the attenuation were formed from is read rather
            // than divided back out -- a washed-out interval leaves nothing to
            // divide by.
            const Scalar bare = replay.entry[row][column];
            carried[row][column] =
                bar_entry[row][column] - bar_recovery[row] * equilibrium[column];
            bar_equilibrium[column] = bar_equilibrium[column]
                - bar_recovery[row] * attenuation * bare;
            bar_attenuation = bar_attenuation + carried[row][column] * bare;
            bar_operator[row][column] = attenuation * carried[row][column];
        }
    }

    Scalar bar_a[3][3]{};
    Scalar bar_third{};
    Scalar bar_minors{};
    Scalar bar_determinant{};
    if (close) {
        Scalar history[SERIES_TERMS][3];
        history[0][0] = Scalar{1.0};
        history[0][1] = Scalar{};
        history[0][2] = Scalar{};
        double factorial = 1.0;
        Scalar weight[SERIES_TERMS];
        weight[0] = Scalar{1.0};
        for (int order = 1; order < SERIES_TERMS; ++order) {
            history[order][0] = history[order - 1][2] * determinant;
            history[order][1] =
                history[order - 1][0] - history[order - 1][2] * minors;
            history[order][2] = history[order - 1][1];
            factorial *= static_cast<double>(order);
            weight[order] = Scalar{1.0 / factorial};
        }
        Scalar sums[3] = {Scalar{}, Scalar{}, Scalar{}};
        for (int order = 0; order < SERIES_TERMS; ++order) {
            for (int slot = 0; slot < 3; ++slot) {
                sums[slot] = sums[slot] + weight[order] * history[order][slot];
            }
        }
        Scalar square[3][3];
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                Scalar total{};
                for (int inner = 0; inner < 3; ++inner) {
                    total = total + shifted[row][inner] * shifted[inner][column];
                }
                square[row][column] = total;
            }
        }
        const Scalar lift = exponential(third);
        Scalar bar_sums[3] = {Scalar{}, Scalar{}, Scalar{}};
        Scalar bar_square[3][3]{};
        Scalar bar_shifted[3][3]{};
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                const Scalar inner = (row == column ? sums[0] : Scalar{})
                    + sums[1] * shifted[row][column]
                    + sums[2] * square[row][column];
                bar_third = bar_third
                    + bar_operator[row][column] * lift * inner;
                const Scalar scaled = bar_operator[row][column] * lift;
                if (row == column) {
                    bar_sums[0] = bar_sums[0] + scaled;
                }
                bar_sums[1] = bar_sums[1] + scaled * shifted[row][column];
                bar_sums[2] = bar_sums[2] + scaled * square[row][column];
                bar_shifted[row][column] =
                    bar_shifted[row][column] + sums[1] * scaled;
                bar_square[row][column] = sums[2] * scaled;
            }
        }
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                for (int inner = 0; inner < 3; ++inner) {
                    bar_shifted[row][inner] = bar_shifted[row][inner]
                        + bar_square[row][column] * shifted[inner][column];
                    bar_shifted[inner][column] = bar_shifted[inner][column]
                        + shifted[row][inner] * bar_square[row][column];
                }
            }
        }
        // The recurrence run backwards, each triple picking up the weight its
        // own term carried into the sums.
        Scalar bar_history[3];
        for (int slot = 0; slot < 3; ++slot) {
            bar_history[slot] = weight[SERIES_TERMS - 1] * bar_sums[slot];
        }
        for (int order = SERIES_TERMS - 1; order > 0; --order) {
            const Scalar held[3] = {
                bar_history[0], bar_history[1], bar_history[2]
            };
            bar_determinant =
                bar_determinant + held[0] * history[order - 1][2];
            bar_minors = bar_minors - held[1] * history[order - 1][2];
            bar_history[0] = weight[order - 1] * bar_sums[0] + held[1];
            bar_history[1] = weight[order - 1] * bar_sums[1] + held[2];
            bar_history[2] = weight[order - 1] * bar_sums[2]
                + held[0] * determinant - held[1] * minors;
        }
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                bar_a[row][column] =
                    bar_a[row][column] + bar_shifted[row][column];
            }
            bar_third = bar_third - bar_shifted[row][row];
        }
    } else {
        const Scalar radius = root(Scalar{-1.0 / 3.0} * minors);
        const Scalar cube = radius * radius * radius;
        const Scalar raw = Scalar{0.5} * determinant * reciprocal(cube);
        constexpr double limit = 1.0 - ARG_GUARD;
        const bool clamped = primal(raw) > limit || primal(raw) < -limit;
        Scalar argument = raw;
        if (primal(raw) > limit) {
            argument = Scalar{limit};
        } else if (primal(raw) < -limit) {
            argument = Scalar{-limit};
        }
        const Scalar angle = Scalar{1.0 / 3.0} * arc_cosine(argument);
        Scalar roots[3];
        int origin[3] = {0, 1, 2};
        for (int turn = 0; turn < 3; ++turn) {
            roots[turn] = Scalar{2.0} * radius
                * cosine(angle - Scalar{TURN_THIRD * turn}) + third;
        }
        for (int pass = 0; pass < 2; ++pass) {
            for (int index = 0; index + 1 < 3; ++index) {
                if (primal(roots[index]) > primal(roots[index + 1])) {
                    const Scalar held = roots[index];
                    roots[index] = roots[index + 1];
                    roots[index + 1] = held;
                    const int seat = origin[index];
                    origin[index] = origin[index + 1];
                    origin[index + 1] = seat;
                }
            }
        }
        const Scalar first = three_pool_difference(roots[0], roots[1]);
        const Scalar upper = three_pool_difference(roots[1], roots[2]);
        const Scalar span = roots[2] - roots[0];
        const Scalar inverse_span = reciprocal(span);
        const Scalar second = (upper - first) * inverse_span;
        const Scalar leading = exponential(roots[0]);

        Scalar bar_leading{};
        Scalar bar_first{};
        Scalar bar_second{};
        Scalar bar_roots[3] = {Scalar{}, Scalar{}, Scalar{}};
        Scalar bar_product[3][3]{};
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                const Scalar from_low = row == column
                    ? a[row][column] - roots[0] : a[row][column];
                Scalar product{};
                for (int inner = 0; inner < 3; ++inner) {
                    const Scalar left = row == inner
                        ? a[row][inner] - roots[0] : a[row][inner];
                    const Scalar right = inner == column
                        ? a[inner][column] - roots[1] : a[inner][column];
                    product = product + left * right;
                }
                const Scalar seed = bar_operator[row][column];
                if (row == column) {
                    bar_leading = bar_leading + seed;
                }
                bar_first = bar_first + seed * from_low;
                bar_second = bar_second + seed * product;
                bar_a[row][column] = bar_a[row][column] + first * seed;
                if (row == column) {
                    bar_roots[0] = bar_roots[0] - first * seed;
                }
                bar_product[row][column] = second * seed;
            }
        }
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                for (int inner = 0; inner < 3; ++inner) {
                    const Scalar left = row == inner
                        ? a[row][inner] - roots[0] : a[row][inner];
                    const Scalar right = inner == column
                        ? a[inner][column] - roots[1] : a[inner][column];
                    const Scalar seed = bar_product[row][column];
                    bar_a[row][inner] = bar_a[row][inner] + seed * right;
                    if (row == inner) {
                        bar_roots[0] = bar_roots[0] - seed * right;
                    }
                    bar_a[inner][column] = bar_a[inner][column] + left * seed;
                    if (inner == column) {
                        bar_roots[1] = bar_roots[1] - left * seed;
                    }
                }
            }
        }
        bar_roots[0] = bar_roots[0] + bar_leading * leading;
        // second = (upper - first) / span
        const Scalar bar_upper = bar_second * inverse_span;
        bar_first = bar_first - bar_second * inverse_span;
        const Scalar bar_span =
            Scalar{} - bar_second * second * inverse_span;
        bar_roots[2] = bar_roots[2] + bar_span;
        bar_roots[0] = bar_roots[0] - bar_span;
        three_pool_difference_adjoint(
            roots[0], roots[1], bar_first, bar_roots[0], bar_roots[1]
        );
        three_pool_difference_adjoint(
            roots[1], roots[2], bar_upper, bar_roots[1], bar_roots[2]
        );

        Scalar bar_radius{};
        Scalar bar_angle{};
        for (int seat = 0; seat < 3; ++seat) {
            const int turn = origin[seat];
            const Scalar swing = angle - Scalar{TURN_THIRD * turn};
            bar_radius = bar_radius + Scalar{2.0} * cosine(swing) * bar_roots[seat];
            bar_angle = bar_angle
                - Scalar{2.0} * radius * sine(swing) * bar_roots[seat];
            bar_third = bar_third + bar_roots[seat];
        }
        if (!clamped) {
            const Scalar slope = Scalar{} - reciprocal(
                Scalar{3.0} * root(Scalar{1.0} - argument * argument)
            );
            const Scalar bar_raw = bar_angle * slope;
            bar_determinant =
                bar_determinant + Scalar{0.5} * bar_raw * reciprocal(cube);
            bar_radius = bar_radius
                - Scalar{3.0} * raw * bar_raw * reciprocal(radius);
        }
        bar_minors = bar_minors
            - bar_radius * reciprocal(Scalar{6.0} * radius);
    }

    // ---- the invariants back onto the shifted generator ----
    Scalar bar_shifted[3][3]{};
    bar_shifted[0][0] = bar_minors * (shifted[1][1] + shifted[2][2]);
    bar_shifted[1][1] = bar_minors * (shifted[0][0] + shifted[2][2]);
    bar_shifted[2][2] = bar_minors * (shifted[0][0] + shifted[1][1]);
    bar_shifted[0][1] = Scalar{} - bar_minors * shifted[1][0];
    bar_shifted[1][0] = Scalar{} - bar_minors * shifted[0][1];
    bar_shifted[0][2] = Scalar{} - bar_minors * shifted[2][0];
    bar_shifted[2][0] = Scalar{} - bar_minors * shifted[0][2];
    bar_shifted[1][2] = Scalar{} - bar_minors * shifted[2][1];
    bar_shifted[2][1] = Scalar{} - bar_minors * shifted[1][2];
    // det = s00(s11 s22 - s12 s21) - s01(s10 s22 - s12 s20)
    //     + s02(s10 s21 - s11 s20)
    bar_shifted[0][0] = bar_shifted[0][0]
        + bar_determinant * (shifted[1][1] * shifted[2][2]
                             - shifted[1][2] * shifted[2][1]);
    bar_shifted[0][1] = bar_shifted[0][1]
        - bar_determinant * (shifted[1][0] * shifted[2][2]
                             - shifted[1][2] * shifted[2][0]);
    bar_shifted[0][2] = bar_shifted[0][2]
        + bar_determinant * (shifted[1][0] * shifted[2][1]
                             - shifted[1][1] * shifted[2][0]);
    bar_shifted[1][0] = bar_shifted[1][0]
        - bar_determinant * (shifted[0][1] * shifted[2][2]
                             - shifted[0][2] * shifted[2][1]);
    bar_shifted[1][1] = bar_shifted[1][1]
        + bar_determinant * (shifted[0][0] * shifted[2][2]
                             - shifted[0][2] * shifted[2][0]);
    bar_shifted[1][2] = bar_shifted[1][2]
        - bar_determinant * (shifted[0][0] * shifted[2][1]
                             - shifted[0][1] * shifted[2][0]);
    bar_shifted[2][0] = bar_shifted[2][0]
        + bar_determinant * (shifted[0][1] * shifted[1][2]
                             - shifted[0][2] * shifted[1][1]);
    bar_shifted[2][1] = bar_shifted[2][1]
        - bar_determinant * (shifted[0][0] * shifted[1][2]
                             - shifted[0][2] * shifted[1][0]);
    bar_shifted[2][2] = bar_shifted[2][2]
        + bar_determinant * (shifted[0][0] * shifted[1][1]
                             - shifted[0][1] * shifted[1][0]);
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            bar_a[row][column] = bar_a[row][column] + bar_shifted[row][column];
        }
        bar_third = bar_third - bar_shifted[row][row];
    }
    for (int row = 0; row < 3; ++row) {
        bar_a[row][row] = bar_a[row][row] + Scalar{1.0 / 3.0} * bar_third;
    }

    // ---- the generator back onto the rates, fractions and interval ----
    const Scalar bar_dt = (Scalar{} - kab - kac - r1_free) * bar_a[0][0]
        + kba * bar_a[0][1] + kca * bar_a[0][2] + kab * bar_a[1][0]
        + (Scalar{} - kba - r1_pool_b) * bar_a[1][1] + kac * bar_a[2][0]
        + (Scalar{} - kca - r1_bound) * bar_a[2][2];
    const Scalar bar_kab = dt * (bar_a[1][0] - bar_a[0][0]);
    const Scalar bar_kba = dt * (bar_a[0][1] - bar_a[1][1]);
    const Scalar bar_kac = dt * (bar_a[2][0] - bar_a[0][0]);
    const Scalar bar_kca = dt * (bar_a[0][2] - bar_a[2][2]);
    Scalar bar_free = bar_equilibrium[0] + exchange_b * bar_kba
        + exchange_c * bar_kca;
    Scalar bar_fraction_b = bar_equilibrium[1] + exchange_b * bar_kab;
    Scalar bar_fraction_c = bar_equilibrium[2] + exchange_c * bar_kac;

    ThreePoolGradient<Scalar> gradient{};
    gradient.r1_free = Scalar{} - dt * bar_a[0][0];
    gradient.r1_pool_b = Scalar{} - dt * bar_a[1][1];
    gradient.r1_bound = Scalar{} - dt * bar_a[2][2];
    gradient.exchange_b = fraction_b * bar_kab + free * bar_kba;
    gradient.exchange_c = fraction_c * bar_kac + free * bar_kca;
    gradient.fraction_b = bar_fraction_b - bar_free;
    gradient.fraction_c = bar_fraction_c - bar_free;
    gradient.dt = bar_dt;
    gradient.attenuation = bar_attenuation;
    return gradient;
}

// The transverse step of two chemically exchanging pools: ``F+ <- E2 F+``
// over one interval, with ``F-`` taking the conjugate of the same operator.
//
// The generator is ``L = K - diag(R2) - 2 pi i diag(df)``, and the closed form
// is the one the longitudinal pair uses -- the numbers have become complex,
// the algebra has not. Two properties make that cheap:
//
//   * The operator is even in the square root. ``cosh(d)`` and ``sinh(d)/d``
//     both are, so it is a function of the discriminant alone and a complex
//     square root's branch cut cannot reach it. No branch is selected here.
//   * ``tau +/- d`` are the eigenvalues, whose real parts are non-positive --
//     an imaginary diagonal is anti-Hermitian and contributes nothing to them
//     -- so forming the pair from their exponentials keeps a long interval
//     from turning an underflow times an overflow into a NaN.
//
// There is no recovery term: transverse magnetization relaxes toward zero.
template <typename Cplx>
struct TwoPoolTransverse {
    Cplx e11;
    Cplx e12;
    Cplx e21;
    Cplx e22;
};

template <typename Real, typename Cplx>
inline TwoPoolTransverse<Cplx> two_pool_transverse_step(
    const Real r2_free,
    const Real r2_bound,
    const Real exchange,
    const Real bound,
    // What the free water is left with, which is not ``1 - bound`` once a
    // semisolid pool holds a share of the voxel too. It carries no transverse
    // magnetization of its own, so it is absent from this 2x2 -- but it is not
    // absent from how much free water the exchange sees.
    const Real free,
    const Real shift_hz,
    const Real dt,
    const Real attenuation
) {
    const Real kab = exchange * bound;
    const Real kba = exchange * free;
    const Cplx l11 = widen((Real{} - kab - r2_free) * dt);
    const Cplx l12 = widen(kba * dt);
    const Cplx l21 = widen(kab * dt);
    // Only pool b's offset appears: pool a sits at whatever off-resonance the
    // free precession already carries the whole voxel through.
    const Cplx l22 = widen((Real{} - kba - r2_bound) * dt)
        - as_imaginary((2.0F * PI) * (shift_hz * dt));

    const Cplx half_trace = 0.5F * (l11 + l22);
    const Cplx half_gap = 0.5F * (l11 - l22);
    const Cplx square = half_gap * half_gap + l12 * l21;
    const Cplx delta = root(square);
    const Cplx upper = exponential(half_trace + delta);
    const Cplx lower = exponential(half_trace - delta);
    const Cplx cosine = 0.5F * (upper + lower);
    // ``sinh(d)/d`` by series near the origin, where the root has no
    // derivative. The threshold is on the discriminant's distance from zero,
    // so both the value and any tangent leave the branch before the division
    // does damage.
    const Cplx scale = primal_norm(square) > 1e-24F
        ? 0.5F * (upper - lower) * reciprocal(delta)
        : exponential(half_trace)
            * (widen(Real{1.0F}) + (1.0F / 6.0F) * square
               + (1.0F / 120.0F) * (square * square));

    TwoPoolTransverse<Cplx> step{};
    step.e11 = attenuation * (cosine + scale * half_gap);
    step.e12 = attenuation * (scale * l12);
    step.e21 = attenuation * (scale * l21);
    step.e22 = attenuation * (cosine - scale * half_gap);
    return step;
}

// What a cotangent on the transverse step leaves on the seven real numbers it
// was formed from.
template <typename Real>
struct TwoPoolTransverseGradient {
    Real r2_free;
    Real r2_bound;
    Real exchange;
    Real bound;
    Real free;
    Real shift_hz;
    Real dt;
    Real attenuation;
};

// The reverse sweep of ``two_pool_transverse_step``.
//
// Every step from the four generator entries to the four operator entries is
// holomorphic, so the sweep is the longitudinal one with complex numbers in
// place of real ones -- no conjugates appear along the way. That holds because
// the cotangents come in as row covectors: ``bar_e`` is the number with
// ``dL = Re(bar_e de)``, which for a caller carrying the usual ``dL =
// Re(conj(z_bar) dz)`` convention means handing over ``conj(z_bar) dz/de``.
// Only where a complex intermediate meets one of the real inputs does a real
// part get taken.
template <typename Real, typename Cplx>
inline TwoPoolTransverseGradient<Real> two_pool_transverse_adjoint(
    const Real r2_free,
    const Real r2_bound,
    const Real exchange,
    const Real bound,
    const Real free,
    const Real shift_hz,
    const Real dt,
    const Real attenuation,
    const Cplx bar_e11,
    const Cplx bar_e12,
    const Cplx bar_e21,
    const Cplx bar_e22
) {
    const Real kab = exchange * bound;
    const Real kba = exchange * free;
    const Cplx l11 = widen((Real{} - kab - r2_free) * dt);
    const Cplx l12 = widen(kba * dt);
    const Cplx l21 = widen(kab * dt);
    const Cplx l22 = widen((Real{} - kba - r2_bound) * dt)
        - as_imaginary((2.0F * PI) * (shift_hz * dt));

    const Cplx half_trace = 0.5F * (l11 + l22);
    const Cplx half_gap = 0.5F * (l11 - l22);
    const Cplx square = half_gap * half_gap + l12 * l21;
    const bool series = !(primal_norm(square) > 1e-24F);
    const Cplx delta = root(square);
    const Cplx upper = exponential(half_trace + delta);
    const Cplx lower = exponential(half_trace - delta);
    const Cplx plain = exponential(half_trace);
    const Cplx cosine = 0.5F * (upper + lower);
    const Cplx scale = series
        ? plain
            * (widen(Real{1.0F}) + (1.0F / 6.0F) * square
               + (1.0F / 120.0F) * (square * square))
        : 0.5F * (upper - lower) * reciprocal(delta);

    const Cplx bare_11 = cosine + scale * half_gap;
    const Cplx bare_12 = scale * l12;
    const Cplx bare_21 = scale * l21;
    const Cplx bare_22 = cosine - scale * half_gap;

    const Cplx bar_attenuation = bar_e11 * bare_11 + bar_e12 * bare_12
        + bar_e21 * bare_21 + bar_e22 * bare_22;
    const Cplx scaled_11 = attenuation * bar_e11;
    const Cplx scaled_12 = attenuation * bar_e12;
    const Cplx scaled_21 = attenuation * bar_e21;
    const Cplx scaled_22 = attenuation * bar_e22;

    const Cplx bar_cosine = scaled_11 + scaled_22;
    const Cplx bar_scale = (scaled_11 - scaled_22) * half_gap
        + scaled_12 * l12 + scaled_21 * l21;
    Cplx bar_half_gap = scale * (scaled_11 - scaled_22);
    Cplx bar_l12 = scale * scaled_12;
    Cplx bar_l21 = scale * scaled_21;

    Cplx bar_half_trace{};
    Cplx bar_square{};
    if (series) {
        bar_half_trace = bar_cosine * cosine + bar_scale * scale;
        bar_square = plain
            * (bar_cosine * (widen(Real{0.5F}) + (1.0F / 12.0F) * square)
               + bar_scale * (widen(Real{1.0F / 6.0F}) + (1.0F / 60.0F) * square));
    } else {
        const Cplx inverse = reciprocal(delta);
        const Cplx bar_upper = 0.5F * (bar_cosine + bar_scale * inverse);
        const Cplx bar_lower = 0.5F * (bar_cosine - bar_scale * inverse);
        bar_half_trace = bar_upper * upper + bar_lower * lower;
        const Cplx bar_delta = bar_upper * upper - bar_lower * lower
            - bar_scale * scale * inverse;
        bar_square = 0.5F * (bar_delta * inverse);
    }

    bar_half_gap = bar_half_gap + 2.0F * (bar_square * half_gap);
    bar_l12 = bar_l12 + bar_square * l21;
    bar_l21 = bar_l21 + bar_square * l12;

    const Cplx bar_l11 = 0.5F * (bar_half_trace + bar_half_gap);
    const Cplx bar_l22 = 0.5F * (bar_half_trace - bar_half_gap);

    const Cplx bar_kab = dt * (bar_l21 - bar_l11);
    const Cplx bar_kba = dt * (bar_l12 - bar_l22);
    const Cplx bar_dt = (Real{} - kab - r2_free) * bar_l11
        + kba * bar_l12 + kab * bar_l21
        + (widen(Real{} - kba - r2_bound)
           - as_imaginary((2.0F * PI) * shift_hz)) * bar_l22;

    TwoPoolTransverseGradient<Real> gradient{};
    gradient.r2_free = Real{} - real_part(dt * bar_l11);
    gradient.r2_bound = Real{} - real_part(dt * bar_l22);
    gradient.exchange = real_part(bound * bar_kab + free * bar_kba);
    gradient.bound = real_part(exchange * bar_kab);
    gradient.free = real_part(exchange * bar_kba);
    gradient.shift_hz = real_part(
        (Real{} - dt) * ((2.0F * PI) * as_imaginary(Real{1.0F}) * bar_l22)
    );
    gradient.dt = real_part(bar_dt);
    gradient.attenuation = real_part(bar_attenuation);
    return gradient;
}

// How well the bound pool absorbs a pulse this far off its resonance, by the
// same cubic Hermite the transition table is read with. Taken in magnitude,
// the lineshape being even, and clamped at the far end: a pulse driven past
// the tabulated offset saturates no less than the last knot says.
inline float lineshape_at(const Buffers& buffers, const float offset_hz) {
    const float last = static_cast<float>(buffers.lineshape_bins - 1);
    const float scaled = std::min(
        std::fabs(offset_hz) / buffers.lineshape_step, last
    );
    const float lower = std::min(std::floor(scaled), last - 1.0F);
    const float u = scaled - lower;
    const float* const near = buffers.lineshape
        + static_cast<std::int64_t>(lower) * LINESHAPE_STRIDE;
    const float u2 = u * u;
    const float u3 = u2 * u;
    const float step = buffers.lineshape_step;
    return (2.0F * u3 - 3.0F * u2 + 1.0F) * near[0]
        + (u3 - 2.0F * u2 + u) * step * near[1]
        + (-2.0F * u3 + 3.0F * u2) * near[2]
        + (u3 - u2) * step * near[3];
}

// The lineshape and its derivative in the *signed* offset, from the same
// cubic. The table covers the magnitude, so the slope changes sign with the
// offset; past the last knot the read is constant and the slope is zero.
inline void lineshape_at_slope(
    const Buffers& buffers, const float offset_hz, float& value, float& slope
) {
    const float last = static_cast<float>(buffers.lineshape_bins - 1);
    const float magnitude = std::fabs(offset_hz) / buffers.lineshape_step;
    const float scaled = std::min(magnitude, last);
    const float lower = std::min(std::floor(scaled), last - 1.0F);
    const float u = scaled - lower;
    const float* const near = buffers.lineshape
        + static_cast<std::int64_t>(lower) * LINESHAPE_STRIDE;
    const float u2 = u * u;
    const float u3 = u2 * u;
    const float step = buffers.lineshape_step;
    value = (2.0F * u3 - 3.0F * u2 + 1.0F) * near[0]
        + (u3 - 2.0F * u2 + u) * step * near[1]
        + (-2.0F * u3 + 3.0F * u2) * near[2]
        + (u3 - u2) * step * near[3];
    if (magnitude > last) {
        slope = 0.0F;
        return;
    }
    const float direction = offset_hz < 0.0F ? -1.0F : 1.0F;
    slope = direction * (
        (6.0F * u2 - 6.0F * u) * near[0] / step
        + (3.0F * u2 - 4.0F * u + 1.0F) * near[1]
        + (-6.0F * u2 + 6.0F * u) * near[2] / step
        + (3.0F * u2 - 2.0F * u) * near[3]
    );
}

// The lineshape, its slope and its curvature, for a pass that differentiates
// the slope in turn. The table covers the magnitude, so the slope changes sign
// with the offset and the curvature does not: an even function's second
// derivative is even.
inline void lineshape_at_curve(
    const Buffers& buffers,
    const float offset_hz,
    float& value,
    float& slope,
    float& curve
) {
    const float last = static_cast<float>(buffers.lineshape_bins - 1);
    const float magnitude = std::fabs(offset_hz) / buffers.lineshape_step;
    const float scaled = std::min(magnitude, last);
    const float lower = std::min(std::floor(scaled), last - 1.0F);
    const float u = scaled - lower;
    const float* const near = buffers.lineshape
        + static_cast<std::int64_t>(lower) * LINESHAPE_STRIDE;
    const float u2 = u * u;
    const float u3 = u2 * u;
    const float step = buffers.lineshape_step;
    value = (2.0F * u3 - 3.0F * u2 + 1.0F) * near[0]
        + (u3 - 2.0F * u2 + u) * step * near[1]
        + (-2.0F * u3 + 3.0F * u2) * near[2]
        + (u3 - u2) * step * near[3];
    if (magnitude > last) {
        slope = 0.0F;
        curve = 0.0F;
        return;
    }
    const float direction = offset_hz < 0.0F ? -1.0F : 1.0F;
    slope = direction * (
        (6.0F * u2 - 6.0F * u) * near[0] / step
        + (3.0F * u2 - 4.0F * u + 1.0F) * near[1]
        + (-6.0F * u2 + 6.0F * u) * near[2] / step
        + (3.0F * u2 - 2.0F * u) * near[3]
    );
    curve = (12.0F * u - 6.0F) * near[0] / (step * step)
        + (6.0F * u - 4.0F) * near[1] / step
        + (-12.0F * u + 6.0F) * near[2] / (step * step)
        + (6.0F * u - 2.0F) * near[3] / step;
}

// The lineshape read along a direction in the offset.
inline DualFloat lineshape_at(const Buffers& buffers, const DualFloat offset_hz) {
    float value = 0.0F;
    float slope = 0.0F;
    lineshape_at_slope(buffers, offset_hz.value, value, slope);
    return DualFloat{value, slope * offset_hz.tangent};
}

// The lineshape and its slope, both carried along a direction in the offset:
// what a pass that differentiates the adjoint needs from the table.
inline void lineshape_at_slope(
    const Buffers& buffers,
    const DualFloat offset_hz,
    DualFloat& value,
    DualFloat& slope
) {
    float shape = 0.0F;
    float first = 0.0F;
    float second = 0.0F;
    lineshape_at_curve(buffers, offset_hz.value, shape, first, second);
    value = DualFloat{shape, first * offset_hz.tangent};
    slope = DualFloat{first, second * offset_hz.tangent};
}

// Where an event's pulse reads the transmit field. A sequence with one shim
// carries a row index of zero on every event, so this is the bare atom index
// for it without the kernel needing to know that in its type.
inline std::int64_t transmit_row(
    const Buffers& buffers, const std::int64_t event, const std::int64_t atom
) {
    return static_cast<std::int64_t>(buffers.shim_index[event])
        * buffers.atom_count + atom;
}

// Whether any atom carries diffusion. The lane kernels keep transcendentals
// out of their state loops, which per-order damping would undo, so they are
// selected only when this is false.
inline bool any_diffusion(const float* rate, const std::int64_t atom_count) {
    for (std::int64_t atom = 0; atom < atom_count; ++atom) {
        if (rate[atom] != 0.0F) {
            return true;
        }
    }
    return false;
}

inline Complex multiply(const Complex left, const Complex right) {
    return left * right;
}

// Where a voxel sits along the slice. Voxels are spread over the profile
// voxel-major, so consecutive atoms walk the slice and wrap.
template <RfMode MODE>
inline std::int64_t slice_row(const Buffers& buffers, const std::int64_t atom) {
    if constexpr (MODE == RfMode::PROFILED) {
        return atom % buffers.locations;
    } else {
        (void)buffers;
        (void)atom;
        return 0;
    }
}

// The rotation a pulse performs at one voxel, read rather than interpolated.
//
// A tabulated pair covers a shape's every pulse because a static array reaches
// the rotation through one complex scalar; this one is integrated per pulse
// per voxel, so there is nothing to interpolate and the read is four floats.
inline std::int64_t dynamic_row(
    const Buffers& buffers, const TrainView& view, const std::int64_t event
) {
    return static_cast<std::int64_t>(
        buffers.dynamic_index[view.event_base + event]
    );
}

inline std::int64_t dynamic_offset(
    const Buffers& buffers, const std::int64_t row, const std::int64_t atom
) {
    return (row * buffers.atom_count + atom) * 4;
}

inline void dynamic_pair_at(
    const Buffers& buffers,
    const std::int64_t row,
    const std::int64_t atom,
    Complex& a,
    Complex& b
) {
    const float* const entry =
        buffers.dynamic + dynamic_offset(buffers, row, atom);
    a = Complex(entry[0], entry[1]);
    b = Complex(entry[2], entry[3]);
}

// The rotation and the direction along it, for a pulse at one voxel. A pass
// that follows no direction passes none, and the rotation is held still.
inline void dynamic_pair_dual_at(
    const Buffers& primal,
    const float* const direction,
    const std::int64_t row,
    const std::int64_t atom,
    DualComplex& a,
    DualComplex& b
) {
    const std::int64_t offset = dynamic_offset(primal, row, atom);
    const float* const value = primal.dynamic + offset;
    Complex moved_a(0.0f, 0.0f);
    Complex moved_b(0.0f, 0.0f);
    if (direction != nullptr) {
        const float* const tangent = direction + offset;
        moved_a = Complex(tangent[0], tangent[1]);
        moved_b = Complex(tangent[2], tangent[3]);
    }
    a = DualComplex{Complex(value[0], value[1]), moved_a};
    b = DualComplex{Complex(value[2], value[3]), moved_b};
}

// Which row of the stacked tables this pulse reads: its own shape's block,
// then the voxel's place along the slice.
template <RfMode MODE>
inline std::int64_t table_row(
    const Buffers& buffers, const std::int64_t event, const std::int64_t location
) {
    if constexpr (MODE == RfMode::PROFILED) {
        return static_cast<std::int64_t>(buffers.profile_index[event])
            * buffers.locations + location;
    } else {
        (void)buffers;
        (void)event;
        (void)location;
        return 0;
    }
}

// The pair the table holds at this flip angle, by cubic Hermite between the
// knots bracketing it. Cubic rather than linear because a linear read has no
// second derivative to give the second-order pass, and because storing the
// slope makes the cubic cost the same two loads.
//
// Clamped at both ends rather than extrapolated: a cubic run off its grid
// leaves the unit circle, and a pulse driven past the tabulated flip should
// saturate rather than diverge.
inline void profile_pair(
    const Buffers& buffers,
    const std::int64_t row,
    const float theta,
    Complex& a,
    Complex& b
) {
    const float last = static_cast<float>(buffers.profile_bins - 1);
    const float scaled = std::min(std::max(theta / buffers.profile_step, 0.0F), last);
    const float lower = std::min(std::floor(scaled), last - 1.0F);
    const float u = scaled - lower;
    const float* const near = buffers.profile
        + (row * buffers.profile_bins + static_cast<std::int64_t>(lower))
            * PROFILE_STRIDE;
    const float* const far = near + PROFILE_STRIDE;

    const float u2 = u * u;
    const float u3 = u2 * u;
    const float h00 = 2.0F * u3 - 3.0F * u2 + 1.0F;
    const float h10 = (u3 - 2.0F * u2 + u) * buffers.profile_step;
    const float h01 = -2.0F * u3 + 3.0F * u2;
    const float h11 = (u3 - u2) * buffers.profile_step;

    a = Complex(
        h00 * near[0] + h10 * near[4] + h01 * far[0] + h11 * far[4],
        h00 * near[1] + h10 * near[5] + h01 * far[1] + h11 * far[5]
    );
    b = Complex(
        h00 * near[2] + h10 * near[6] + h01 * far[2] + h11 * far[6],
        h00 * near[3] + h10 * near[7] + h01 * far[3] + h11 * far[7]
    );
}

inline void shift(std::vector<Complex>& fplus, std::vector<Complex>& fminus) {
    const std::size_t count = fplus.size();
    for (std::size_t state = 0; state + 1 < count; ++state) {
        fminus[state] = fminus[state + 1];
    }
    fminus[count - 1] = Complex{};
    for (std::size_t state = count - 1; state > 0; --state) {
        fplus[state] = fplus[state - 1];
    }
    fplus[0] = std::conj(fminus[0]);
}

inline void rotate(
    std::vector<Complex>& fplus,
    std::vector<Complex>& fminus,
    std::vector<Complex>& longitudinal,
    const float alpha,
    const float phi
) {
    const float cosine = std::cos(alpha);
    const float sine = std::sin(alpha);
    const float cosine_half_sq = 0.5F * (1.0F + cosine);
    const float sine_half_sq = 0.5F * (1.0F - cosine);
    const Complex phase_one = std::polar(1.0F, phi);
    const Complex phase_two = phase_one * phase_one;
    const Complex t00(cosine_half_sq, 0.0F);
    const Complex t01 = sine_half_sq * phase_two;
    const Complex t02 = Complex(0.0F, -sine) * phase_one;
    const Complex t10 = std::conj(t01);
    const Complex t11 = t00;
    const Complex t12 = Complex(0.0F, sine) * std::conj(phase_one);
    const Complex t20 = Complex(0.0F, -0.5F * sine) * std::conj(phase_one);
    const Complex t21 = Complex(0.0F, 0.5F * sine) * phase_one;
    const Complex t22(cosine, 0.0F);

    for (std::size_t state = 0; state < fplus.size(); ++state) {
        const Complex fp = fplus[state];
        const Complex fm = fminus[state];
        const Complex z = longitudinal[state];
        fplus[state] = multiply(t00, fp) + multiply(t01, fm) + multiply(t02, z);
        fminus[state] = multiply(t10, fp) + multiply(t11, fm) + multiply(t12, z);
        longitudinal[state] =
            multiply(t20, fp) + multiply(t21, fm) + multiply(t22, z);
    }
}

// The pair and its derivative in the flip angle, from the same cubic. The
// derivative of a Hermite segment is another polynomial in the same four knot
// values, so reading both costs one extra combination rather than a second
// table.
//
// Past the ends the value is clamped and the derivative is that of the clamped
// curve, which is the last segment's -- callers refuse a pulse that reaches
// there, so this only has to stay finite.
inline void profile_pair_slope(
    const Buffers& buffers,
    const std::int64_t row,
    const float theta,
    Complex& a,
    Complex& b,
    Complex& slope_a,
    Complex& slope_b
) {
    const float last = static_cast<float>(buffers.profile_bins - 1);
    const float scaled = std::min(std::max(theta / buffers.profile_step, 0.0F), last);
    const float lower = std::min(std::floor(scaled), last - 1.0F);
    const float u = scaled - lower;
    const float* const near = buffers.profile
        + (row * buffers.profile_bins + static_cast<std::int64_t>(lower))
            * PROFILE_STRIDE;
    const float* const far = near + PROFILE_STRIDE;

    const float u2 = u * u;
    const float u3 = u2 * u;
    const float step = buffers.profile_step;
    const float h00 = 2.0F * u3 - 3.0F * u2 + 1.0F;
    const float h10 = (u3 - 2.0F * u2 + u) * step;
    const float h01 = -2.0F * u3 + 3.0F * u2;
    const float h11 = (u3 - u2) * step;
    // d/dtheta is d/du over the knot spacing.
    const float g00 = (6.0F * u2 - 6.0F * u) / step;
    const float g10 = 3.0F * u2 - 4.0F * u + 1.0F;
    const float g01 = (-6.0F * u2 + 6.0F * u) / step;
    const float g11 = 3.0F * u2 - 2.0F * u;

    float value[4];
    float slope[4];
    for (std::size_t part = 0; part < 4; ++part) {
        value[part] = h00 * near[part] + h10 * near[part + 4]
            + h01 * far[part] + h11 * far[part + 4];
        slope[part] = g00 * near[part] + g10 * near[part + 4]
            + g01 * far[part] + g11 * far[part + 4];
    }
    a = Complex(value[0], value[1]);
    b = Complex(value[2], value[3]);
    slope_a = Complex(slope[0], slope[1]);
    slope_b = Complex(slope[2], slope[3]);
}

// The rotation a pulse of any shape performs, named by its Cayley-Klein pair:
//
//   T = [ conj(a)^2   -conj(b)^2   -2 conj(a b) ]
//       [ -b^2         a^2         -2 a b       ]
//       [ conj(a) b    a conj(b)   |a|^2-|b|^2  ]
//
// `rotate` above is the case of an instantaneous pulse, kept separate rather
// than folded into this: it reaches the same matrix through different
// arithmetic, and a sequence with no table must not move in the last place for
// the sake of one that has a table.
inline void rotate_spinor(
    std::vector<Complex>& fplus,
    std::vector<Complex>& fminus,
    std::vector<Complex>& longitudinal,
    const Complex a,
    const Complex b
) {
    const Complex conj_a = std::conj(a);
    const Complex conj_b = std::conj(b);
    const Complex t00 = conj_a * conj_a;
    const Complex t01 = -conj_b * conj_b;
    const Complex t02 = -2.0F * std::conj(a * b);
    const Complex t10 = -b * b;
    const Complex t11 = a * a;
    const Complex t12 = -2.0F * a * b;
    const Complex t20 = conj_a * b;
    const Complex t21 = a * conj_b;
    const Complex t22(std::norm(a) - std::norm(b), 0.0F);

    for (std::size_t state = 0; state < fplus.size(); ++state) {
        const Complex fp = fplus[state];
        const Complex fm = fminus[state];
        const Complex z = longitudinal[state];
        fplus[state] = multiply(t00, fp) + multiply(t01, fm) + multiply(t02, z);
        fminus[state] = multiply(t10, fp) + multiply(t11, fm) + multiply(t12, z);
        longitudinal[state] =
            multiply(t20, fp) + multiply(t21, fm) + multiply(t22, z);
    }
}

inline DualComplex apply(
    const DualCoefficient coefficient,
    const DualComplex state
) {
    return {
        coefficient.value * state.value,
        coefficient.tangent * state.value + coefficient.value * state.tangent,
    };
}

inline DualComplex add(
    const DualComplex first,
    const DualComplex second,
    const DualComplex third
) {
    return {
        first.value + second.value + third.value,
        first.tangent + second.tangent + third.tangent,
    };
}

// The spinor rotation carrying a forward-mode tangent. Every entry of the
// matrix is a product of two factors drawn from the pair and its conjugate, so
// its tangent is the same product differentiated once.
inline void rotate_spinor(
    std::vector<DualComplex>& fplus,
    std::vector<DualComplex>& fminus,
    std::vector<DualComplex>& longitudinal,
    const DualComplex a,
    const DualComplex b
) {
    const DualComplex conj_a{std::conj(a.value), std::conj(a.tangent)};
    const DualComplex conj_b{std::conj(b.value), std::conj(b.tangent)};

    auto product = [](const DualComplex left, const DualComplex right) {
        return DualCoefficient{
            left.value * right.value,
            left.tangent * right.value + left.value * right.tangent,
        };
    };
    auto scaled = [](const DualCoefficient term, const float factor) {
        return DualCoefficient{factor * term.value, factor * term.tangent};
    };

    const DualCoefficient aa = product(conj_a, conj_a);
    const DualCoefficient bb = product(conj_b, conj_b);
    const DualCoefficient ab = product(conj_a, conj_b);
    const DualCoefficient plain_bb = product(b, b);
    const DualCoefficient plain_aa = product(a, a);
    const DualCoefficient plain_ab = product(a, b);

    const DualCoefficient t00 = aa;
    const DualCoefficient t01 = scaled(bb, -1.0F);
    const DualCoefficient t02 = scaled(ab, -2.0F);
    const DualCoefficient t10 = scaled(plain_bb, -1.0F);
    const DualCoefficient t11 = plain_aa;
    const DualCoefficient t12 = scaled(plain_ab, -2.0F);
    const DualCoefficient t20 = product(conj_a, b);
    const DualCoefficient t21 = product(a, conj_b);
    const DualCoefficient norm_a = product(a, conj_a);
    const DualCoefficient norm_b = product(b, conj_b);
    const DualCoefficient t22{
        norm_a.value - norm_b.value, norm_a.tangent - norm_b.tangent
    };

    for (std::size_t state = 0; state < fplus.size(); ++state) {
        const DualComplex fp = fplus[state];
        const DualComplex fm = fminus[state];
        const DualComplex z = longitudinal[state];
        fplus[state] = add(apply(t00, fp), apply(t01, fm), apply(t02, z));
        fminus[state] = add(apply(t10, fp), apply(t11, fm), apply(t12, z));
        longitudinal[state] =
            add(apply(t20, fp), apply(t21, fm), apply(t22, z));
    }
}

inline void shift(
    std::vector<DualComplex>& fplus,
    std::vector<DualComplex>& fminus
) {
    const std::size_t count = fplus.size();
    for (std::size_t state = 0; state + 1 < count; ++state) {
        fminus[state] = fminus[state + 1];
    }
    fminus[count - 1] = DualComplex{};
    for (std::size_t state = count - 1; state > 0; --state) {
        fplus[state] = fplus[state - 1];
    }
    fplus[0] = {
        std::conj(fminus[0].value),
        std::conj(fminus[0].tangent),
    };
}

inline void rotate(
    std::vector<DualComplex>& fplus,
    std::vector<DualComplex>& fminus,
    std::vector<DualComplex>& longitudinal,
    const float alpha,
    const float alpha_tangent,
    const float phi,
    const float phi_tangent
) {
    const float cosine = std::cos(alpha);
    const float sine = std::sin(alpha);
    const float cosine_tangent = -sine * alpha_tangent;
    const float sine_tangent = cosine * alpha_tangent;
    const float cosine_half_sq = 0.5F * (1.0F + cosine);
    const float sine_half_sq = 0.5F * (1.0F - cosine);
    const float cosine_half_tangent = 0.5F * cosine_tangent;
    const float sine_half_tangent = -0.5F * cosine_tangent;
    const Complex phase_one = std::polar(1.0F, phi);
    const Complex phase_one_tangent = Complex(0.0F, phi_tangent) * phase_one;
    const Complex phase_two = phase_one * phase_one;
    const Complex phase_two_tangent =
        Complex(0.0F, 2.0F * phi_tangent) * phase_two;

    const DualCoefficient t00{
        Complex(cosine_half_sq, 0.0F),
        Complex(cosine_half_tangent, 0.0F),
    };
    const DualCoefficient t01{
        sine_half_sq * phase_two,
        sine_half_tangent * phase_two + sine_half_sq * phase_two_tangent,
    };
    const DualCoefficient t02{
        Complex(0.0F, -sine) * phase_one,
        Complex(0.0F, -sine_tangent) * phase_one
            + Complex(0.0F, -sine) * phase_one_tangent,
    };
    const DualCoefficient t10{std::conj(t01.value), std::conj(t01.tangent)};
    const DualCoefficient t11 = t00;
    const DualCoefficient t12{
        Complex(0.0F, sine) * std::conj(phase_one),
        Complex(0.0F, sine_tangent) * std::conj(phase_one)
            + Complex(0.0F, sine) * std::conj(phase_one_tangent),
    };
    const DualCoefficient t20{
        Complex(0.0F, -0.5F * sine) * std::conj(phase_one),
        Complex(0.0F, -0.5F * sine_tangent) * std::conj(phase_one)
            + Complex(0.0F, -0.5F * sine) * std::conj(phase_one_tangent),
    };
    const DualCoefficient t21{
        Complex(0.0F, 0.5F * sine) * phase_one,
        Complex(0.0F, 0.5F * sine_tangent) * phase_one
            + Complex(0.0F, 0.5F * sine) * phase_one_tangent,
    };
    const DualCoefficient t22{
        Complex(cosine, 0.0F),
        Complex(cosine_tangent, 0.0F),
    };

    for (std::size_t state = 0; state < fplus.size(); ++state) {
        const DualComplex fp = fplus[state];
        const DualComplex fm = fminus[state];
        const DualComplex z = longitudinal[state];
        fplus[state] = add(apply(t00, fp), apply(t01, fm), apply(t02, z));
        fminus[state] = add(apply(t10, fp), apply(t11, fm), apply(t12, z));
        longitudinal[state] =
            add(apply(t20, fp), apply(t21, fm), apply(t22, z));
    }
}

// Inlined into the vector clones below rather than called from them: a
// clone is a copy of the caller, so a body reached through a call is
// compiled once for the baseline instruction set and no more.
template <RfMode MODE, Pools POOLS>
TORCHSIM_ALWAYS_INLINE void simulate_jvp_range(
    const JvpBuffers& buffers,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count
) {
    constexpr bool MT = POOLS == Pools::SEMISOLID;
    constexpr bool BM = POOLS == Pools::EXCHANGING;
    constexpr bool THREE = POOLS == Pools::THREE;
    constexpr bool TWO_POOL = MT || BM;
    // Which pool carries a transverse pair of its own, and which run absorbs
    // the power a pulse deposits; see ``simulate_range``.
    constexpr bool PAIRED = BM || THREE;
    constexpr bool SATURATED = MT || THREE;
    const Buffers& primal = buffers.primal;
    const bool flowing = primal.moving;
    const std::size_t states = static_cast<std::size_t>(state_count);
    std::vector<DualComplex> fplus(states);
    std::vector<DualComplex> fminus(states);
    std::vector<DualComplex> longitudinal(states);
    std::vector<DualComplex> bound((TWO_POOL || THREE) ? states : 0U);
    std::vector<DualComplex> semisolid(THREE ? states : 0U);
    std::vector<DualComplex> bound_plus(PAIRED ? states : 0U);
    std::vector<DualComplex> bound_minus(PAIRED ? states : 0U);
    Damping<DualFloat> damping(states);

    for (std::int64_t work = work_begin; work < work_end; ++work) {
        const TrainView view = train_view(primal, work, event_count, output_count);
        const std::int64_t atom = view.atom;
        const std::int64_t location = slice_row<MODE>(primal, atom);
        const float* const dot_duration = buffers.duration + view.event_base;
        const float* const dot_flip = buffers.flip + view.event_base;
        const float* const dot_phase = buffers.phase + view.event_base;
        std::fill(fplus.begin(), fplus.end(), DualComplex{});
        std::fill(fminus.begin(), fminus.end(), DualComplex{});
        std::fill(longitudinal.begin(), longitudinal.end(), DualComplex{});
        // Equilibrium is split between the pools, so a direction along the
        // bound fraction moves magnetization from one to the other before a
        // single event has run.
        const DualFloat bound_fraction = (BM || THREE)
            ? DualFloat{
                primal.pool_b_fraction[atom], buffers.pool_b_fraction[atom]
            }
            : (MT
                ? DualFloat{
                    primal.bound_fraction[atom], buffers.bound_fraction[atom]
                }
                : DualFloat{});
        const DualFloat semisolid_fraction = SATURATED
            ? DualFloat{primal.bound_fraction[atom], buffers.bound_fraction[atom]}
            : DualFloat{};
        const DualFloat held_free = THREE
            ? DualFloat{1.0F, 0.0F} - bound_fraction - semisolid_fraction
            : DualFloat{1.0F, 0.0F} - bound_fraction;
        longitudinal[0] = DualComplex{
            Complex(held_free.value, 0.0F), Complex(held_free.tangent, 0.0F)
        };
        if constexpr (TWO_POOL || THREE) {
            std::fill(bound.begin(), bound.end(), DualComplex{});
            bound[0] = DualComplex{
                Complex(bound_fraction.value, 0.0F),
                Complex(bound_fraction.tangent, 0.0F),
            };
        }
        if constexpr (THREE) {
            std::fill(semisolid.begin(), semisolid.end(), DualComplex{});
            semisolid[0] = DualComplex{
                Complex(semisolid_fraction.value, 0.0F),
                Complex(semisolid_fraction.tangent, 0.0F),
            };
        }
        if constexpr (PAIRED) {
            std::fill(bound_plus.begin(), bound_plus.end(), DualComplex{});
            std::fill(bound_minus.begin(), bound_minus.end(), DualComplex{});
        }

        const float t1 = primal.t1[atom];
        const float t2 = primal.t2[atom];
        const float r1 = 1000.0F / t1;
        const float r2 = 1000.0F / t2;
        // A rate is the reciprocal of a time, so a direction along the time
        // arrives at the rate with the sign turned over and the square beneath.
        const DualFloat rate_free{
            r1, -1000.0F * buffers.t1[atom] / (t1 * t1)
        };
        const DualFloat rate_bound = (BM || THREE)
            ? DualFloat{
                1000.0F / primal.t1_pool_b[atom],
                -1000.0F * buffers.t1_pool_b[atom]
                    / (primal.t1_pool_b[atom] * primal.t1_pool_b[atom]),
            }
            : (MT
                ? DualFloat{
                    1000.0F / primal.t1_bound[atom],
                    -1000.0F * buffers.t1_bound[atom]
                        / (primal.t1_bound[atom] * primal.t1_bound[atom]),
                }
                : DualFloat{});
        const DualFloat rate_semisolid = SATURATED
            ? DualFloat{
                1000.0F / primal.t1_bound[atom],
                -1000.0F * buffers.t1_bound[atom]
                    / (primal.t1_bound[atom] * primal.t1_bound[atom]),
            }
            : DualFloat{};
        const DualFloat rate2_bound = PAIRED
            ? DualFloat{
                1000.0F / primal.t2_pool_b[atom],
                -1000.0F * buffers.t2_pool_b[atom]
                    / (primal.t2_pool_b[atom] * primal.t2_pool_b[atom]),
            }
            : DualFloat{};
        const DualFloat rate2_free{
            r2, -1000.0F * buffers.t2[atom] / (t2 * t2)
        };
        const DualFloat pool_shift = PAIRED
            ? DualFloat{primal.pool_b_shift[atom], buffers.pool_b_shift[atom]}
            : DualFloat{};
        const DualFloat exchange = (BM || THREE)
            ? DualFloat{
                primal.pool_b_exchange[atom], buffers.pool_b_exchange[atom]
            }
            : (MT
                ? DualFloat{
                    primal.bound_exchange[atom], buffers.exchange_rate[atom]
                }
                : DualFloat{});
        const DualFloat semisolid_exchange = SATURATED
            ? DualFloat{primal.bound_exchange[atom], buffers.exchange_rate[atom]}
            : DualFloat{};
        const DualFloat transverse_free = THREE
            ? DualFloat{1.0F, 0.0F} - bound_fraction - semisolid_fraction
            : DualFloat{1.0F, 0.0F} - bound_fraction;
        const DualFloat damping_rate = held_rate(
            primal.diffusion, buffers.diffusion, atom, primal.diffusing
        );
        // With one shim the transmit field is a property of the voxel and
        // lifts out of the event loop; with several it belongs to the shim a
        // pulse drives, and is read where the pulse is.
        const bool shimmed = primal.shim_count > 1;
        const float voxel_b1 = primal.transmit ? primal.b1[atom] : 1.0F;
        const float voxel_dot_b1 = primal.transmit ? buffers.b1[atom] : 0.0F;
        const float voxel_b1_phase =
            primal.off_axis ? primal.b1_phase[atom] : 0.0F;
        const float voxel_dot_b1_phase =
            primal.off_axis ? buffers.b1_phase[atom] : 0.0F;
        const float velocity = flowing ? primal.velocity[atom] : 0.0F;
        const float dot_velocity = flowing ? buffers.velocity[atom] : 0.0F;
        const DualFloat flow_rate{
            velocity * primal.flow_scale, dot_velocity * primal.flow_scale
        };
        const DualFloat washout_rate{
            std::fabs(velocity) * primal.washout_scale,
            speed_direction(velocity) * dot_velocity * primal.washout_scale,
        };
        for (std::int64_t event = 0; event < event_count; ++event) {
            const float dt = view.duration[event];
            const float dt_tangent = dot_duration[event];
            const DualFloat interval{dt, dt_tangent};
            damping.set(damping_rate, interval);
            const DualFloat wout = washout_out(washout_rate, interval);
            const float dry1 = std::exp(-r1 * dt);
            const float dry2 = std::exp(-r2 * dt);
            const float e1 = dry1 * wout.value;
            const float e2 = dry2 * wout.value;
            const float e1_tangent = e1 * (
                1000.0F * dt * buffers.t1[atom] / (t1 * t1)
                - r1 * dt_tangent
            ) + dry1 * wout.tangent;
            const float e2_tangent = e2 * (
                1000.0F * dt * buffers.t2[atom] / (t2 * t2)
                - r2 * dt_tangent
            ) + dry2 * wout.tangent;
            const float angle =
                primal.off_axis ? -2.0F * PI * primal.b0[atom] * dt : 0.0F;
            const float angle_tangent = primal.off_axis
                ? -2.0F * PI * (
                    buffers.b0[atom] * dt + primal.b0[atom] * dt_tangent
                )
                : 0.0F;
            const Complex phase = turned(1.0F, angle, primal.off_axis);
            const Complex phase_tangent = Complex(0.0F, angle_tangent) * phase;
            const Complex off_resonance = e2 * phase;
            const Complex off_resonance_tangent =
                e2_tangent * phase + e2 * phase_tangent;
            // The exchange operator belongs to the interval, not to a dephasing
            // order, so it is formed once and carries its own tangent; the
            // per-order damping multiplies both.
            // Three pools mix through a 3x3 formed in double and handed back
            // narrowed; see ``three_pool_step``.
            const ThreePoolStep<DualDouble> triple = THREE
                ? three_pool_step<DualDouble>(
                    widen_dual(rate_free), widen_dual(rate_bound),
                    widen_dual(rate_semisolid), widen_dual(exchange),
                    widen_dual(semisolid_exchange), widen_dual(bound_fraction),
                    widen_dual(semisolid_fraction), widen_dual(interval),
                    widen_dual(wout)
                )
                : ThreePoolStep<DualDouble>{};
            const TwoPoolStep<DualFloat> pools = TWO_POOL
                ? two_pool_step(
                    rate_free, rate_bound, exchange, bound_fraction, interval, wout
                )
                : TwoPoolStep<DualFloat>{};
            const TwoPoolTransverse<DualComplex> across = PAIRED
                ? two_pool_transverse_step<DualFloat, DualComplex>(
                    rate2_free, rate2_bound, exchange, bound_fraction,
                    transverse_free, pool_shift, interval, wout
                )
                : TwoPoolTransverse<DualComplex>{};
            for (std::int64_t state = 0; state < state_count; ++state) {
                const std::size_t index = static_cast<std::size_t>(state);
                const DualFloat damp_transverse = damping.transverse[index];
                const DualFloat damp_longitudinal = damping.longitudinal[index];
                DualComplex& fp = fplus[index];
                DualComplex& fm = fminus[index];
                DualComplex& z = longitudinal[index];
                DualFloat turn_longitudinal{};
                DualFloat turn_transverse{};
                if (flowing) {
                    flow_turn_dual(
                        flow_rate, interval, index, turn_longitudinal,
                        turn_transverse
                    );
                }
                const Complex spin_transverse =
                    turned(1.0F, turn_transverse.value, flowing);
                const Complex spin_transverse_tangent =
                    Complex(0.0F, turn_transverse.tangent) * spin_transverse;
                const Complex spin_longitudinal =
                    turned(1.0F, turn_longitudinal.value, flowing);
                const Complex spin_longitudinal_tangent =
                    Complex(0.0F, turn_longitudinal.tangent) * spin_longitudinal;
                const Complex damped = off_resonance * damp_transverse.value;
                const Complex damped_tangent =
                    off_resonance_tangent * damp_transverse.value
                    + off_resonance * damp_transverse.tangent;
                const Complex off = damped * spin_transverse;
                const Complex off_tangent =
                    damped_tangent * spin_transverse + damped * spin_transverse_tangent;
                const Complex conjugate_off = std::conj(off);
                const Complex conjugate_tangent = std::conj(off_tangent);
                if constexpr (PAIRED) {
                    // The exchange operator carries the chemical shift; what
                    // is left is the off-resonance and damping both pools take
                    // alike, which is ``off`` without its own relaxation.
                    const DualComplex carried{
                        off / e2,
                        (off_tangent - (e2_tangent / e2) * off) / e2,
                    };
                    DualComplex& bp = bound_plus[index];
                    DualComplex& bm = bound_minus[index];
                    const DualComplex free_plus = fp;
                    const DualComplex held_plus = bp;
                    const DualComplex free_minus = fm;
                    const DualComplex held_minus = bm;
                    fp = (across.e11 * free_plus + across.e12 * held_plus)
                        * carried;
                    bp = (across.e21 * free_plus + across.e22 * held_plus)
                        * carried;
                    // ``F-`` follows the conjugate of the operator entry by
                    // entry, not its transpose.
                    const DualComplex conjugated = conjugate(carried);
                    fm = (conjugate(across.e11) * free_minus
                        + conjugate(across.e12) * held_minus) * conjugated;
                    bm = (conjugate(across.e21) * free_minus
                        + conjugate(across.e22) * held_minus) * conjugated;
                } else {
                    fp.tangent = fp.tangent * off + fp.value * off_tangent;
                    fp.value *= off;
                    fm.tangent =
                        fm.tangent * conjugate_off + fm.value * conjugate_tangent;
                    fm.value *= conjugate_off;
                }
                // Both pools take the same per-order damping and phase: their
                // order-n states describe one dephasing configuration, and the
                // bound pool has no diffusion coefficient of its own.
                const DualComplex spin{
                    damp_longitudinal.value * spin_longitudinal,
                    damp_longitudinal.tangent * spin_longitudinal
                        + damp_longitudinal.value * spin_longitudinal_tangent,
                };
                if constexpr (THREE) {
                    DualComplex& held = bound[index];
                    DualComplex& stuck = semisolid[index];
                    const DualComplex pools_in[3] = {z, held, stuck};
                    DualComplex mixed[3];
                    for (int row = 0; row < 3; ++row) {
                        DualComplex carried{};
                        for (int column = 0; column < 3; ++column) {
                            carried = carried
                                + narrow_dual(triple.entry[row][column])
                                    * pools_in[column];
                        }
                        mixed[row] = carried * spin;
                    }
                    z = mixed[0];
                    held = mixed[1];
                    stuck = mixed[2];
                } else if constexpr (TWO_POOL) {
                    DualComplex& held = bound[index];
                    const DualComplex free_state = z;
                    const DualComplex bound_state = held;
                    z = (pools.e11 * free_state + pools.e12 * bound_state) * spin;
                    held =
                        (pools.e21 * free_state + pools.e22 * bound_state) * spin;
                } else {
                    const float recovered = e1 * damp_longitudinal.value;
                    const float recovered_tangent =
                        e1_tangent * damp_longitudinal.value
                        + e1 * damp_longitudinal.tangent;
                    const Complex relaxation = recovered * spin_longitudinal;
                    const Complex relaxation_tangent =
                        recovered_tangent * spin_longitudinal
                        + recovered * spin_longitudinal_tangent;
                    z.tangent =
                        z.tangent * relaxation + z.value * relaxation_tangent;
                    z.value *= relaxation;
                }
            }
            if constexpr (THREE) {
                DualComplex* const seats[3] = {
                    &longitudinal[0], &bound[0], &semisolid[0]
                };
                for (int row = 0; row < 3; ++row) {
                    const DualFloat grown = narrow_dual(triple.recovery[row]);
                    seats[row]->value += Complex(grown.value, 0.0F);
                    seats[row]->tangent += Complex(grown.tangent, 0.0F);
                }
            } else if constexpr (TWO_POOL) {
                longitudinal[0].value += Complex(pools.recovery_free.value, 0.0F);
                longitudinal[0].tangent +=
                    Complex(pools.recovery_free.tangent, 0.0F);
                bound[0].value += Complex(pools.recovery_bound.value, 0.0F);
                bound[0].tangent += Complex(pools.recovery_bound.tangent, 0.0F);
            } else {
                longitudinal[0].value += Complex(1.0F - e1, 0.0F);
                longitudinal[0].tangent -= Complex(e1_tangent, 0.0F);
            }

            const std::uint8_t action = primal.action[event];
            if ((action & PRE_SHIFT) != 0) {
                shift(fplus, fminus);
                if constexpr (PAIRED) {
                    shift(bound_plus, bound_minus);
                }
            }
            if (primal.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    // Only the free pool; see the forward kernel for why.
                    const float efficiency = -scaled(
                        1.0F, primal.inversion_efficiency[atom], primal.inverting
                    );
                    const float efficiency_tangent = primal.inverting
                        ? -buffers.inversion_efficiency[atom]
                        : 0.0F;
                    for (DualComplex& value : longitudinal) {
                        value.tangent = efficiency_tangent * value.value
                            + efficiency * value.tangent;
                        value.value *= efficiency;
                    }
                    if constexpr (PAIRED) {
                        for (DualComplex& value : bound) {
                            value.tangent = efficiency_tangent * value.value
                                + efficiency * value.tangent;
                            value.value *= efficiency;
                        }
                    }
                } else {
                    const std::int64_t transmit =
                        shimmed ? transmit_row(primal, event, atom) : atom;
                    const float pulse_b1 =
                        shimmed ? primal.b1[transmit] : voxel_b1;
                    const float pulse_dot_b1 =
                        shimmed ? buffers.b1[transmit] : voxel_dot_b1;
                    const float alpha = view.flip[event] * pulse_b1;
                    const float alpha_tangent = dot_flip[event] * pulse_b1
                        + view.flip[event] * pulse_dot_b1;
                    const float phi = view.phase[event]
                        + (shimmed ? primal.b1_phase[transmit] : voxel_b1_phase);
                    const float phi_tangent = dot_phase[event]
                        + (shimmed ? buffers.b1_phase[transmit]
                                   : voxel_dot_b1_phase);
                    if constexpr (SATURATED) {
                        // The semisolid pool absorbs the power the pulse deposits,
                        // so it reads the bare flip the transmit field gives
                        // the voxel. The offset reaches it through the voxel's
                        // own off-resonance, which is where the lineshape's
                        // slope enters a forward direction.
                        const float offset =
                            primal.rf_frequency[event] - primal.b0[atom];
                        float shape = 0.0F;
                        float shape_slope = 0.0F;
                        lineshape_at_slope(primal, offset, shape, shape_slope);
                        const float deposited = primal.saturation[event];
                        const DualFloat exponent{
                            deposited * alpha * alpha * shape,
                            deposited * (
                                2.0F * alpha * alpha_tangent * shape
                                - alpha * alpha * shape_slope * buffers.b0[atom]
                            ),
                        };
                        const DualFloat absorbed = dual_exp(exponent);
                        for (DualComplex& value : (THREE ? semisolid : bound)) {
                            value.tangent = absorbed.tangent * value.value
                                + absorbed.value * value.tangent;
                            value.value *= absorbed.value;
                        }
                    }
                    if constexpr (MODE != RfMode::INSTANT) {
                        const Complex turn = std::polar(1.0F, -phi);
                        DualComplex shaped_a{};
                        DualComplex shaped_b{};
                        if constexpr (MODE == RfMode::DYNAMIC) {
                            // The array was resolved outside the kernel, so a
                            // direction along it arrives already carried
                            // through the pulse integral. Only the phase is
                            // left to turn the axis by.
                            DualComplex pair_a{};
                            DualComplex pair_b{};
                            dynamic_pair_dual_at(
                                primal, buffers.dynamic,
                                dynamic_row(primal, view, event),
                                atom, pair_a, pair_b
                            );
                            const Complex spun = pair_b.value * turn;
                            shaped_a = pair_a;
                            shaped_b = DualComplex{
                                spun,
                                pair_b.tangent * turn
                                    - Complex(0.0F, phi_tangent) * spun,
                            };
                        } else {
                            Complex pair_a{};
                            Complex pair_b{};
                            Complex slope_a{};
                            Complex slope_b{};
                            profile_pair_slope(
                                primal, table_row<MODE>(primal, event, location),
                                alpha, pair_a, pair_b, slope_a, slope_b
                            );
                            // The flip angle carries the tangent into the
                            // table; the RF phase turns the axis after it
                            // comes out.
                            const Complex spun = pair_b * turn;
                            shaped_a = DualComplex{
                                pair_a, slope_a * alpha_tangent
                            };
                            shaped_b = DualComplex{
                                spun,
                                slope_b * turn * alpha_tangent
                                    - Complex(0.0F, phi_tangent) * spun,
                            };
                        }
                        rotate_spinor(
                            fplus, fminus, longitudinal, shaped_a, shaped_b
                        );
                        if constexpr (PAIRED) {
                            rotate_spinor(
                                bound_plus, bound_minus, bound, shaped_a,
                                shaped_b
                            );
                        }
                    } else {
                        rotate(
                            fplus,
                            fminus,
                            longitudinal,
                            alpha,
                            alpha_tangent,
                            phi,
                            phi_tangent
                        );
                        if constexpr (PAIRED) {
                            rotate(
                                bound_plus,
                                bound_minus,
                                bound,
                                alpha,
                                alpha_tangent,
                                phi,
                                phi_tangent
                            );
                        }
                    }
                }
            } else if (primal.kind[event] == 2 && (action & RECORD) != 0) {
                const std::int64_t output = primal.output_index[event];
                const Complex demodulation = std::polar(1.0F, -view.phase[event]);
                const Complex demodulation_tangent =
                    Complex(0.0F, -dot_phase[event]) * demodulation;
                const DualComplex fp = PAIRED
                    ? DualComplex{
                        fplus[0].value + bound_plus[0].value,
                        fplus[0].tangent + bound_plus[0].tangent,
                    }
                    : fplus[0];
                const float density = primal.density ? primal.m0[atom] : 1.0F;
                const float dot_density =
                    primal.density ? buffers.m0[atom] : 0.0F;
                const Complex signal_tangent =
                    dot_density * fp.value * demodulation
                    + density * fp.tangent * demodulation
                    + density * fp.value * demodulation_tangent;
                const std::int64_t index = view.output_base + output;
                primal.output_real[index] = signal_tangent.real();
                primal.output_imag[index] = signal_tangent.imag();
            }
            if ((action & POST_SHIFT) != 0) {
                shift(fplus, fminus);
                if constexpr (PAIRED) {
                    shift(bound_plus, bound_minus);
                }
            }
            if ((action & SPOIL_AFTER) != 0) {
                std::fill(fplus.begin(), fplus.end(), DualComplex{});
                std::fill(fminus.begin(), fminus.end(), DualComplex{});
                if constexpr (PAIRED) {
                    std::fill(
                        bound_plus.begin(), bound_plus.end(), DualComplex{}
                    );
                    std::fill(
                        bound_minus.begin(), bound_minus.end(), DualComplex{}
                    );
                }
            } else if ((action & SHIFT_AFTER) != 0) {
                shift(fplus, fminus);
                if constexpr (PAIRED) {
                    shift(bound_plus, bound_minus);
                }
            }
        }
    }
}

#if TORCHSIM_MULTIVERSIONED
__attribute__((target_clones("default", "sse4.2", "avx2", "avx512f")))
#endif
inline void shift_real(
    std::vector<float>& plus, std::vector<float>& minus, const std::size_t states
) {
    // No states is impossible -- every entry point rejects a state count
    // below one -- but not provable here, and the loops below count down
    // from ``states - 1``.
    if (states == 0) {
        return;
    }
    for (std::size_t state = 0; state + 1 < states; ++state) {
        minus[state] = minus[state + 1];
    }
    minus[states - 1] = 0.0F;
    for (std::size_t state = states - 1; state > 0; --state) {
        plus[state] = plus[state - 1];
    }
    plus[0] = -minus[0];
}

// ---------------------------------------------------------------------------
// Real-subspace forward.
//
// When every refocusing pulse shares a phase and there is no off-resonance or
// transmit phase, writing F+ = e^{i phi} i a, F- = e^{-i phi} i b, Z = c leaves
// a, b and c real for the whole train: relaxation scales them, the shift's
// conjugate coupling becomes a0 = -b0, and the RF rotation reduces to a real
// 3x3 in the flip angle alone. The recorded sample is then i * m0 * a0.
//
// Callers must have established those conditions; see real_subspace_axis.
// ---------------------------------------------------------------------------

void simulate_real_range(
    const Buffers& buffers,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count
) {
    const std::size_t states = static_cast<std::size_t>(state_count);
    std::vector<float> plus(states);
    std::vector<float> minus(states);
    std::vector<float> longitudinal(states);
    Damping<float> damping(states);

    for (std::int64_t work = work_begin; work < work_end; ++work) {
        const TrainView view = train_view(buffers, work, event_count, output_count);
        const std::int64_t atom = view.atom;
        std::fill(plus.begin(), plus.end(), 0.0F);
        std::fill(minus.begin(), minus.end(), 0.0F);
        std::fill(longitudinal.begin(), longitudinal.end(), 0.0F);
        longitudinal[0] = 1.0F;

        const float r1 = 1000.0F / buffers.t1[atom];
        const float r2 = 1000.0F / buffers.t2[atom];
        const float b1 = buffers.transmit ? buffers.b1[atom] : 1.0F;
        const float m0 = buffers.density ? buffers.m0[atom] : 1.0F;
        const float damping_rate =
            buffers.diffusing ? buffers.diffusion[atom] : 0.0F;

        for (std::int64_t event = 0; event < event_count; ++event) {
            const float dt = view.duration[event];
            damping.set(damping_rate, dt);
            const float e1 = std::exp(-r1 * dt);
            const float e2 = std::exp(-r2 * dt);
            for (std::size_t state = 0; state < states; ++state) {
                const float damp_transverse = e2 * damping.transverse[state];
                plus[state] *= damp_transverse;
                minus[state] *= damp_transverse;
                longitudinal[state] *= e1 * damping.longitudinal[state];
            }
            longitudinal[0] += 1.0F - e1;

            const std::uint8_t action = buffers.action[event];
            if ((action & PRE_SHIFT) != 0) {
                shift_real(plus, minus, states);
            }
            if (buffers.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    const float efficiency = -scaled(
                        1.0F, buffers.inversion_efficiency[atom], buffers.inverting
                    );
                    for (std::size_t state = 0; state < states; ++state) {
                        longitudinal[state] *= efficiency;
                    }
                } else {
                    const float alpha = view.flip[event] * b1;
                    const float cosine = std::cos(alpha);
                    const float sine = std::sin(alpha);
                    const float cosine_half_sq = 0.5F * (1.0F + cosine);
                    const float sine_half_sq = 0.5F * (1.0F - cosine);
                    const float half_sine = 0.5F * sine;
                    for (std::size_t state = 0; state < states; ++state) {
                        const float p = plus[state];
                        const float m = minus[state];
                        const float z = longitudinal[state];
                        plus[state] = cosine_half_sq * p + sine_half_sq * m - sine * z;
                        minus[state] = sine_half_sq * p + cosine_half_sq * m + sine * z;
                        longitudinal[state] = half_sine * p - half_sine * m + cosine * z;
                    }
                }
            } else if (buffers.kind[event] == 2 && (action & RECORD) != 0) {
                const std::int64_t index =
                    view.output_base + buffers.output_index[event];
                buffers.output_real[index] = 0.0F;
                buffers.output_imag[index] = m0 * plus[0];
            }
            if ((action & POST_SHIFT) != 0) {
                shift_real(plus, minus, states);
            }
            if ((action & SPOIL_AFTER) != 0) {
                std::fill(plus.begin(), plus.end(), 0.0F);
                std::fill(minus.begin(), minus.end(), 0.0F);
            } else if ((action & SHIFT_AFTER) != 0) {
                shift_real(plus, minus, states);
            }
        }
    }
}

// Forward-mode through the real subspace. The tangent obeys the same closure as
// the value, so a, b and c each carry a derivative and the rotation is the same
// real 3x3 differentiated by the product rule.
//
// A tangent along b0, b1_phase or an RF phase would leave the subspace; callers
// must seed only directions that stay inside it.
void simulate_real_jvp_range(
    const JvpBuffers& buffers,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count
) {
    const Buffers& primal = buffers.primal;
    const std::size_t states = static_cast<std::size_t>(state_count);
    std::vector<float> plus(states), minus(states), longitudinal(states);
    std::vector<float> dot_plus(states), dot_minus(states), dot_longitudinal(states);
    Damping<DualFloat> damping(states);

    for (std::int64_t work = work_begin; work < work_end; ++work) {
        const TrainView view = train_view(primal, work, event_count, output_count);
        const std::int64_t atom = view.atom;
        const float* const dot_duration = buffers.duration + view.event_base;
        const float* const dot_flip = buffers.flip + view.event_base;

        std::fill(plus.begin(), plus.end(), 0.0F);
        std::fill(minus.begin(), minus.end(), 0.0F);
        std::fill(longitudinal.begin(), longitudinal.end(), 0.0F);
        std::fill(dot_plus.begin(), dot_plus.end(), 0.0F);
        std::fill(dot_minus.begin(), dot_minus.end(), 0.0F);
        std::fill(dot_longitudinal.begin(), dot_longitudinal.end(), 0.0F);
        longitudinal[0] = 1.0F;

        const float t1 = primal.t1[atom];
        const float t2 = primal.t2[atom];
        const float r1 = 1000.0F / t1;
        const float r2 = 1000.0F / t2;
        const float b1 = primal.transmit ? primal.b1[atom] : 1.0F;
        const float m0 = primal.density ? primal.m0[atom] : 1.0F;
        const float dot_t1 = buffers.t1[atom];
        const float dot_t2 = buffers.t2[atom];
        const float dot_b1 = primal.transmit ? buffers.b1[atom] : 0.0F;
        const float dot_m0 = buffers.m0[atom];
        const DualFloat damping_rate = held_rate(
            primal.diffusion, buffers.diffusion, atom, primal.diffusing
        );

        for (std::int64_t event = 0; event < event_count; ++event) {
            const float dt = view.duration[event];
            const float dt_tangent = dot_duration[event];
            damping.set(damping_rate, DualFloat{dt, dt_tangent});
            const float e1 = std::exp(-r1 * dt);
            const float e2 = std::exp(-r2 * dt);
            const float e1_tangent =
                e1 * (1000.0F * dt * dot_t1 / (t1 * t1) - r1 * dt_tangent);
            const float e2_tangent =
                e2 * (1000.0F * dt * dot_t2 / (t2 * t2) - r2 * dt_tangent);
            for (std::size_t state = 0; state < states; ++state) {
                const DualFloat damp_transverse = damping.transverse[state];
                const DualFloat damp_longitudinal = damping.longitudinal[state];
                const float transverse = e2 * damp_transverse.value;
                const float transverse_tangent = e2_tangent * damp_transverse.value
                    + e2 * damp_transverse.tangent;
                const float relaxation = e1 * damp_longitudinal.value;
                const float relaxation_tangent = e1_tangent * damp_longitudinal.value
                    + e1 * damp_longitudinal.tangent;
                dot_plus[state] =
                    dot_plus[state] * transverse + plus[state] * transverse_tangent;
                plus[state] *= transverse;
                dot_minus[state] =
                    dot_minus[state] * transverse + minus[state] * transverse_tangent;
                minus[state] *= transverse;
                dot_longitudinal[state] = dot_longitudinal[state] * relaxation
                    + longitudinal[state] * relaxation_tangent;
                longitudinal[state] *= relaxation;
            }
            longitudinal[0] += 1.0F - e1;
            dot_longitudinal[0] -= e1_tangent;

            const std::uint8_t action = primal.action[event];
            if ((action & PRE_SHIFT) != 0) {
                shift_real(plus, minus, states);
                shift_real(dot_plus, dot_minus, states);
            }
            if (primal.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    const float efficiency = -scaled(
                        1.0F, primal.inversion_efficiency[atom], primal.inverting
                    );
                    const float efficiency_tangent = primal.inverting
                        ? -buffers.inversion_efficiency[atom]
                        : 0.0F;
                    for (std::size_t state = 0; state < states; ++state) {
                        dot_longitudinal[state] =
                            dot_longitudinal[state] * efficiency
                            + longitudinal[state] * efficiency_tangent;
                        longitudinal[state] *= efficiency;
                    }
                } else {
                    const float alpha = view.flip[event] * b1;
                    const float alpha_tangent =
                        dot_flip[event] * b1 + view.flip[event] * dot_b1;
                    const float cosine = std::cos(alpha);
                    const float sine = std::sin(alpha);
                    const float cosine_half_sq = 0.5F * (1.0F + cosine);
                    const float sine_half_sq = 0.5F * (1.0F - cosine);
                    const float half_sine = 0.5F * sine;
                    const float cosine_tangent = -sine * alpha_tangent;
                    const float sine_tangent = cosine * alpha_tangent;
                    const float cosine_half_sq_tangent = -0.5F * sine * alpha_tangent;
                    const float sine_half_sq_tangent = 0.5F * sine * alpha_tangent;
                    const float half_sine_tangent = 0.5F * cosine * alpha_tangent;
                    for (std::size_t state = 0; state < states; ++state) {
                        const float p = plus[state];
                        const float m = minus[state];
                        const float z = longitudinal[state];
                        const float dp = dot_plus[state];
                        const float dm = dot_minus[state];
                        const float dz = dot_longitudinal[state];
                        dot_plus[state] = cosine_half_sq * dp + cosine_half_sq_tangent * p
                            + sine_half_sq * dm + sine_half_sq_tangent * m
                            - sine * dz - sine_tangent * z;
                        dot_minus[state] = sine_half_sq * dp + sine_half_sq_tangent * p
                            + cosine_half_sq * dm + cosine_half_sq_tangent * m
                            + sine * dz + sine_tangent * z;
                        dot_longitudinal[state] = half_sine * dp + half_sine_tangent * p
                            - half_sine * dm - half_sine_tangent * m
                            + cosine * dz + cosine_tangent * z;
                        plus[state] = cosine_half_sq * p + sine_half_sq * m - sine * z;
                        minus[state] = sine_half_sq * p + cosine_half_sq * m + sine * z;
                        longitudinal[state] = half_sine * p - half_sine * m + cosine * z;
                    }
                }
            } else if (primal.kind[event] == 2 && (action & RECORD) != 0) {
                const std::int64_t index =
                    view.output_base + primal.output_index[event];
                primal.output_real[index] = 0.0F;
                primal.output_imag[index] = dot_m0 * plus[0] + m0 * dot_plus[0];
            }
            if ((action & POST_SHIFT) != 0) {
                shift_real(plus, minus, states);
                shift_real(dot_plus, dot_minus, states);
            }
            if ((action & SPOIL_AFTER) != 0) {
                std::fill(plus.begin(), plus.end(), 0.0F);
                std::fill(minus.begin(), minus.end(), 0.0F);
                std::fill(dot_plus.begin(), dot_plus.end(), 0.0F);
                std::fill(dot_minus.begin(), dot_minus.end(), 0.0F);
            } else if ((action & SHIFT_AFTER) != 0) {
                shift_real(plus, minus, states);
                shift_real(dot_plus, dot_minus, states);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Train-lane vectorization of the real-subspace kernels.
//
// A block of REAL_LANES trains shares one atom, so every per-atom quantity is
// uniform across lanes while duration and flip vary with the train. States live
// as [state][lane] with the lane axis innermost and contiguous, which turns each
// state loop into packed arithmetic over independent trains. The lane axis
// carries no reduction: the gradient sums that run across states in the scalar
// kernels become one running total per lane, so nothing forces a horizontal add
// inside the loop.
//
// A partial block repeats its first train into the unused lanes so the
// arithmetic stays uniform; those lanes are not written back.
// ---------------------------------------------------------------------------

constexpr std::size_t REAL_LANES = 8;

// Setting TORCHSIM_REAL_SCALAR=1 routes the real-subspace path through the
// one-train-at-a-time kernels instead. They compute the same thing, so this
// selects between two implementations of one contract: useful for timing one
// against the other, and as a way out if a host miscompiles the lane arithmetic.
inline bool lane_kernels_enabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("TORCHSIM_REAL_SCALAR");
        return value == nullptr || value[0] != '1';
    }();
    return enabled;
}

// A lane loop has a constant trip count, so the compiler unrolls it and then
// declines to re-form the pieces into packed arithmetic. Spelling the width out
// as a vector type settles it, and costs nothing where the extension is absent.
#if defined(__GNUC__) || defined(__clang__)
// A 32-byte vector crosses a function boundary differently depending on
// whether the caller was built with AVX, and GCC says so. Every function that
// returns one here is inline and internal to this file, so no call ever spans
// two translation units and the two conventions never meet.
// The inline bodies are emitted at the end of the translation unit, which is
// where the diagnostic lands, so this stays open rather than being popped.
#pragma GCC diagnostic ignored "-Wpsabi"
using LaneVector = float __attribute__((vector_size(sizeof(float) * REAL_LANES)));
#else
struct LaneVector {
    float lane[REAL_LANES];
};

inline LaneVector operator+(const LaneVector left, const LaneVector right) {
    LaneVector out{};
    for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
        out.lane[lane] = left.lane[lane] + right.lane[lane];
    }
    return out;
}

inline LaneVector operator-(const LaneVector left, const LaneVector right) {
    LaneVector out{};
    for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
        out.lane[lane] = left.lane[lane] - right.lane[lane];
    }
    return out;
}

inline LaneVector operator*(const LaneVector left, const LaneVector right) {
    LaneVector out{};
    for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
        out.lane[lane] = left.lane[lane] * right.lane[lane];
    }
    return out;
}

inline LaneVector operator-(const LaneVector value) {
    LaneVector out{};
    for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
        out.lane[lane] = -value.lane[lane];
    }
    return out;
}
#endif

// memcpy rather than a cast: the state planes carry only float alignment, and
// this is the spelling every compiler turns into a single unaligned move.
inline LaneVector lane_load(const float* const source) {
    LaneVector out;
    std::memcpy(&out, source, sizeof(out));
    return out;
}

inline void lane_store(float* const target, const LaneVector value) {
    std::memcpy(target, &value, sizeof(value));
}

inline LaneVector lane_splat(const float value) {
    float buffer[REAL_LANES];
    for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
        buffer[lane] = value;
    }
    return lane_load(buffer);
}

struct LaneView {
    std::int64_t atom;
    std::int64_t train_begin;
    std::int64_t active;
};

inline std::int64_t lane_blocks(const std::int64_t train_count) {
    const std::int64_t lanes = static_cast<std::int64_t>(REAL_LANES);
    return (train_count + lanes - 1) / lanes;
}

inline LaneView lane_view(const Buffers& buffers, const std::int64_t work) {
    const std::int64_t atom = work % buffers.atom_count;
    const std::int64_t begin =
        (work / buffers.atom_count) * static_cast<std::int64_t>(REAL_LANES);
    return LaneView{
        atom,
        begin,
        std::min<std::int64_t>(
            static_cast<std::int64_t>(REAL_LANES), buffers.train_count - begin
        ),
    };
}

// One event's value for every lane, repeating the block's first train into the
// inactive lanes of a partial block.
inline void gather_lanes(
    const float* buffer,
    const LaneView& view,
    const std::int64_t event,
    const std::int64_t event_count,
    float* out
) {
    for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
        const std::int64_t offset =
            static_cast<std::int64_t>(lane) < view.active
                ? static_cast<std::int64_t>(lane)
                : 0;
        out[lane] = buffer[(view.train_begin + offset) * event_count + event];
    }
}

inline void shift_real_lanes(float* plus, float* minus, const std::size_t states) {
    // No states is impossible -- every entry point rejects a state count
    // below one -- but not provable here, and the loops below count down
    // from ``states - 1``.
    if (states == 0) {
        return;
    }
    for (std::size_t state = 0; state + 1 < states; ++state) {
        lane_store(
            minus + state * REAL_LANES, lane_load(minus + (state + 1) * REAL_LANES)
        );
    }
    lane_store(minus + (states - 1) * REAL_LANES, lane_splat(0.0F));
    for (std::size_t state = states - 1; state > 0; --state) {
        lane_store(
            plus + state * REAL_LANES, lane_load(plus + (state - 1) * REAL_LANES)
        );
    }
    lane_store(plus, -lane_load(minus));
}

// Lanes over atoms: one train's events, eight tissues at a time. A dictionary
// carries one train and many atoms, which is the way round a block of trains
// cannot fill -- and it is the cheaper way round besides, since a tissue
// property is contiguous in the atom index while an event value is one number
// every lane shares.
struct AtomLaneView {
    std::int64_t train;
    std::int64_t atom_begin;
    std::int64_t active;
};

inline AtomLaneView atom_lane_view(const Buffers& buffers, const std::int64_t work) {
    const std::int64_t blocks = lane_blocks(buffers.atom_count);
    const std::int64_t train = work / blocks;
    const std::int64_t begin =
        (work % blocks) * static_cast<std::int64_t>(REAL_LANES);
    return AtomLaneView{
        train,
        begin,
        std::min<std::int64_t>(
            static_cast<std::int64_t>(REAL_LANES), buffers.atom_count - begin
        ),
    };
}

// One tissue property for every lane, repeating the block's first atom into the
// inactive lanes of a partial block. A property the run does not declare reads
// as its identity in every lane.
inline void gather_atoms(
    const float* const buffer,
    const bool present,
    const float identity,
    const AtomLaneView& view,
    float* const out
) {
    if (!present || buffer == nullptr) {
        for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
            out[lane] = identity;
        }
        return;
    }
    for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
        const std::int64_t offset =
            static_cast<std::int64_t>(lane) < view.active
                ? static_cast<std::int64_t>(lane)
                : 0;
        out[lane] = buffer[view.atom_begin + offset];
    }
}

// The forward-mode pass of the real subspace, eight atoms at a time. The state
// arithmetic is the per-train kernel's; what differs is where a lane comes
// from, so the tissue is gathered once per block and every event value is a
// splat.
void simulate_real_jvp_atom_lane_range(
    const JvpBuffers& buffers,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count
) {
    const Buffers& primal = buffers.primal;
    const std::size_t states = static_cast<std::size_t>(state_count);
    const std::size_t width = states * REAL_LANES;
    std::vector<float> storage(6U * width);
    float* const TORCHSIM_RESTRICT plus = storage.data();
    float* const TORCHSIM_RESTRICT minus = plus + width;
    float* const TORCHSIM_RESTRICT longitudinal = minus + width;
    float* const TORCHSIM_RESTRICT dot_plus = longitudinal + width;
    float* const TORCHSIM_RESTRICT dot_minus = dot_plus + width;
    float* const TORCHSIM_RESTRICT dot_longitudinal = dot_minus + width;

    float t1[REAL_LANES], t2[REAL_LANES], dot_t1[REAL_LANES], dot_t2[REAL_LANES];
    float rate1[REAL_LANES], rate2[REAL_LANES];
    float t1_scale[REAL_LANES], t2_scale[REAL_LANES];
    float b1[REAL_LANES], dot_b1[REAL_LANES];
    float m0[REAL_LANES], dot_m0[REAL_LANES];
    float efficiency[REAL_LANES], efficiency_dot[REAL_LANES];
    float e1[REAL_LANES], e2[REAL_LANES];
    float e1_dot[REAL_LANES], e2_dot[REAL_LANES];
    float cosine[REAL_LANES], sine[REAL_LANES], alpha_dot[REAL_LANES];
    const LaneVector half = lane_splat(0.5F);
    const LaneVector one = lane_splat(1.0F);

    for (std::int64_t work = work_begin; work < work_end; ++work) {
        const AtomLaneView view = atom_lane_view(primal, work);
        const std::int64_t event_base = view.train * event_count;
        std::fill(storage.begin(), storage.end(), 0.0F);
        for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
            longitudinal[lane] = 1.0F;
        }

        gather_atoms(primal.t1, true, 1.0F, view, t1);
        gather_atoms(primal.t2, true, 1.0F, view, t2);
        gather_atoms(buffers.t1, true, 0.0F, view, dot_t1);
        gather_atoms(buffers.t2, true, 0.0F, view, dot_t2);
        gather_atoms(primal.b1, primal.transmit, 1.0F, view, b1);
        gather_atoms(buffers.b1, primal.transmit, 0.0F, view, dot_b1);
        gather_atoms(primal.m0, primal.density, 1.0F, view, m0);
        gather_atoms(buffers.m0, primal.density, 0.0F, view, dot_m0);
        gather_atoms(
            primal.inversion_efficiency, primal.inverting, 1.0F, view, efficiency
        );
        gather_atoms(
            buffers.inversion_efficiency, primal.inverting, 0.0F, view, efficiency_dot
        );
        for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
            rate1[lane] = 1000.0F / t1[lane];
            rate2[lane] = 1000.0F / t2[lane];
            t1_scale[lane] = 1000.0F * dot_t1[lane] / (t1[lane] * t1[lane]);
            t2_scale[lane] = 1000.0F * dot_t2[lane] / (t2[lane] * t2[lane]);
            // An inversion turns the longitudinal states over, so the factor
            // the kernel applies is the negative of the efficiency.
            efficiency[lane] = -efficiency[lane];
            efficiency_dot[lane] = -efficiency_dot[lane];
        }

        for (std::int64_t event = 0; event < event_count; ++event) {
            const float dt = primal.duration[event_base + event];
            const float dt_dot = buffers.duration[event_base + event];
            for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
                e1[lane] = std::exp(-rate1[lane] * dt);
                e2[lane] = std::exp(-rate2[lane] * dt);
                e1_dot[lane] = e1[lane] * (dt * t1_scale[lane] - rate1[lane] * dt_dot);
                e2_dot[lane] = e2[lane] * (dt * t2_scale[lane] - rate2[lane] * dt_dot);
            }
            const LaneVector v_e1 = lane_load(e1);
            const LaneVector v_e2 = lane_load(e2);
            const LaneVector v_e1_dot = lane_load(e1_dot);
            const LaneVector v_e2_dot = lane_load(e2_dot);
            for (std::size_t state = 0; state < states; ++state) {
                const std::size_t base = state * REAL_LANES;
                const LaneVector p = lane_load(plus + base);
                const LaneVector m = lane_load(minus + base);
                const LaneVector z = lane_load(longitudinal + base);
                lane_store(
                    dot_plus + base, lane_load(dot_plus + base) * v_e2 + p * v_e2_dot
                );
                lane_store(plus + base, p * v_e2);
                lane_store(
                    dot_minus + base, lane_load(dot_minus + base) * v_e2 + m * v_e2_dot
                );
                lane_store(minus + base, m * v_e2);
                lane_store(
                    dot_longitudinal + base,
                    lane_load(dot_longitudinal + base) * v_e1 + z * v_e1_dot
                );
                lane_store(longitudinal + base, z * v_e1);
            }
            lane_store(longitudinal, lane_load(longitudinal) + (one - v_e1));
            lane_store(dot_longitudinal, lane_load(dot_longitudinal) - v_e1_dot);

            const std::uint8_t action = primal.action[event];
            if ((action & PRE_SHIFT) != 0) {
                shift_real_lanes(plus, minus, states);
                shift_real_lanes(dot_plus, dot_minus, states);
            }
            if (primal.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    const LaneVector factor = lane_load(efficiency);
                    const LaneVector factor_dot = lane_load(efficiency_dot);
                    for (std::size_t state = 0; state < states; ++state) {
                        const std::size_t base = state * REAL_LANES;
                        const LaneVector z = lane_load(longitudinal + base);
                        lane_store(
                            dot_longitudinal + base,
                            lane_load(dot_longitudinal + base) * factor + z * factor_dot
                        );
                        lane_store(longitudinal + base, z * factor);
                    }
                } else {
                    const float flip = primal.flip[event_base + event];
                    const float flip_dot = buffers.flip[event_base + event];
                    for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
                        const float alpha = flip * b1[lane];
                        cosine[lane] = std::cos(alpha);
                        sine[lane] = std::sin(alpha);
                        alpha_dot[lane] = flip_dot * b1[lane] + flip * dot_b1[lane];
                    }
                    const LaneVector c = lane_load(cosine);
                    const LaneVector s = lane_load(sine);
                    const LaneVector rate = lane_load(alpha_dot);
                    const LaneVector chs = half * (one + c);
                    const LaneVector shs = half * (one - c);
                    const LaneVector hs = half * s;
                    const LaneVector c_dot = -(s * rate);
                    const LaneVector s_dot = c * rate;
                    const LaneVector chs_dot = half * c_dot;
                    const LaneVector shs_dot = -(half * c_dot);
                    const LaneVector hs_dot = half * s_dot;
                    for (std::size_t state = 0; state < states; ++state) {
                        const std::size_t base = state * REAL_LANES;
                        const LaneVector p = lane_load(plus + base);
                        const LaneVector m = lane_load(minus + base);
                        const LaneVector z = lane_load(longitudinal + base);
                        const LaneVector dp = lane_load(dot_plus + base);
                        const LaneVector dm = lane_load(dot_minus + base);
                        const LaneVector dz = lane_load(dot_longitudinal + base);
                        lane_store(
                            dot_plus + base,
                            chs * dp + chs_dot * p + shs * dm + shs_dot * m
                                - s * dz - s_dot * z
                        );
                        lane_store(
                            dot_minus + base,
                            shs * dp + shs_dot * p + chs * dm + chs_dot * m
                                + s * dz + s_dot * z
                        );
                        lane_store(
                            dot_longitudinal + base,
                            hs * dp + hs_dot * p - hs * dm - hs_dot * m
                                + c * dz + c_dot * z
                        );
                        lane_store(plus + base, chs * p + shs * m - s * z);
                        lane_store(minus + base, shs * p + chs * m + s * z);
                        lane_store(longitudinal + base, hs * p - hs * m + c * z);
                    }
                }
            } else if (primal.kind[event] == 2 && (action & RECORD) != 0) {
                const std::int64_t offset = primal.output_index[event];
                for (std::int64_t lane = 0; lane < view.active; ++lane) {
                    const std::int64_t index =
                        (view.train * primal.atom_count + view.atom_begin + lane)
                            * output_count
                        + offset;
                    primal.output_real[index] = 0.0F;
                    primal.output_imag[index] =
                        dot_m0[lane] * plus[lane] + m0[lane] * dot_plus[lane];
                }
            }
            if ((action & POST_SHIFT) != 0) {
                shift_real_lanes(plus, minus, states);
                shift_real_lanes(dot_plus, dot_minus, states);
            }
            if ((action & SPOIL_AFTER) != 0) {
                std::fill(plus, plus + width, 0.0F);
                std::fill(minus, minus + width, 0.0F);
                std::fill(dot_plus, dot_plus + width, 0.0F);
                std::fill(dot_minus, dot_minus + width, 0.0F);
            } else if ((action & SHIFT_AFTER) != 0) {
                shift_real_lanes(plus, minus, states);
                shift_real_lanes(dot_plus, dot_minus, states);
            }
        }
    }
}

void simulate_real_jvp_lane_range(
    const JvpBuffers& buffers,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count
) {
    const Buffers& primal = buffers.primal;
    const std::size_t states = static_cast<std::size_t>(state_count);
    const std::size_t width = states * REAL_LANES;
    std::vector<float> storage(6U * width);
    float* const TORCHSIM_RESTRICT plus = storage.data();
    float* const TORCHSIM_RESTRICT minus = plus + width;
    float* const TORCHSIM_RESTRICT longitudinal = minus + width;
    float* const TORCHSIM_RESTRICT dot_plus = longitudinal + width;
    float* const TORCHSIM_RESTRICT dot_minus = dot_plus + width;
    float* const TORCHSIM_RESTRICT dot_longitudinal = dot_minus + width;

    float dt[REAL_LANES], dt_dot[REAL_LANES];
    float flip[REAL_LANES], flip_dot[REAL_LANES];
    float e1[REAL_LANES], e2[REAL_LANES];
    float e1_dot[REAL_LANES], e2_dot[REAL_LANES];
    float cosine[REAL_LANES], sine[REAL_LANES], alpha_dot[REAL_LANES];
    const LaneVector half = lane_splat(0.5F);
    const LaneVector one = lane_splat(1.0F);

    for (std::int64_t work = work_begin; work < work_end; ++work) {
        const LaneView view = lane_view(primal, work);
        const std::int64_t atom = view.atom;
        std::fill(storage.begin(), storage.end(), 0.0F);
        for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
            longitudinal[lane] = 1.0F;
        }

        const float t1 = primal.t1[atom];
        const float t2 = primal.t2[atom];
        const float r1 = 1000.0F / t1;
        const float r2 = 1000.0F / t2;
        const float b1 = primal.transmit ? primal.b1[atom] : 1.0F;
        const float m0 = primal.density ? primal.m0[atom] : 1.0F;
        const float dot_b1 = primal.transmit ? buffers.b1[atom] : 0.0F;
        const float dot_m0 = primal.density ? buffers.m0[atom] : 0.0F;
        const float t1_scale = 1000.0F * buffers.t1[atom] / (t1 * t1);
        const float t2_scale = 1000.0F * buffers.t2[atom] / (t2 * t2);

        for (std::int64_t event = 0; event < event_count; ++event) {
            gather_lanes(primal.duration, view, event, event_count, dt);
            gather_lanes(buffers.duration, view, event, event_count, dt_dot);
            for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
                e1[lane] = std::exp(-r1 * dt[lane]);
                e2[lane] = std::exp(-r2 * dt[lane]);
                e1_dot[lane] = e1[lane] * (dt[lane] * t1_scale - r1 * dt_dot[lane]);
                e2_dot[lane] = e2[lane] * (dt[lane] * t2_scale - r2 * dt_dot[lane]);
            }
            const LaneVector v_e1 = lane_load(e1);
            const LaneVector v_e2 = lane_load(e2);
            const LaneVector v_e1_dot = lane_load(e1_dot);
            const LaneVector v_e2_dot = lane_load(e2_dot);
            for (std::size_t state = 0; state < states; ++state) {
                const std::size_t base = state * REAL_LANES;
                const LaneVector p = lane_load(plus + base);
                const LaneVector m = lane_load(minus + base);
                const LaneVector z = lane_load(longitudinal + base);
                lane_store(
                    dot_plus + base,
                    lane_load(dot_plus + base) * v_e2 + p * v_e2_dot
                );
                lane_store(plus + base, p * v_e2);
                lane_store(
                    dot_minus + base,
                    lane_load(dot_minus + base) * v_e2 + m * v_e2_dot
                );
                lane_store(minus + base, m * v_e2);
                lane_store(
                    dot_longitudinal + base,
                    lane_load(dot_longitudinal + base) * v_e1 + z * v_e1_dot
                );
                lane_store(longitudinal + base, z * v_e1);
            }
            lane_store(longitudinal, lane_load(longitudinal) + (one - v_e1));
            lane_store(dot_longitudinal, lane_load(dot_longitudinal) - v_e1_dot);

            const std::uint8_t action = primal.action[event];
            if ((action & PRE_SHIFT) != 0) {
                shift_real_lanes(plus, minus, states);
                shift_real_lanes(dot_plus, dot_minus, states);
            }
            if (primal.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    const float efficiency = -scaled(
                        1.0F, primal.inversion_efficiency[atom], primal.inverting
                    );
                    const float efficiency_dot = primal.inverting
                        ? -buffers.inversion_efficiency[atom]
                        : 0.0F;
                    for (std::size_t slot = 0; slot < width; ++slot) {
                        dot_longitudinal[slot] = dot_longitudinal[slot] * efficiency
                            + longitudinal[slot] * efficiency_dot;
                        longitudinal[slot] *= efficiency;
                    }
                } else {
                    gather_lanes(primal.flip, view, event, event_count, flip);
                    gather_lanes(buffers.flip, view, event, event_count, flip_dot);
                    for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
                        const float alpha = flip[lane] * b1;
                        cosine[lane] = std::cos(alpha);
                        sine[lane] = std::sin(alpha);
                        alpha_dot[lane] = flip_dot[lane] * b1 + flip[lane] * dot_b1;
                    }
                    const LaneVector c = lane_load(cosine);
                    const LaneVector s = lane_load(sine);
                    const LaneVector rate = lane_load(alpha_dot);
                    const LaneVector chs = half * (one + c);
                    const LaneVector shs = half * (one - c);
                    const LaneVector hs = half * s;
                    const LaneVector c_dot = -(s * rate);
                    const LaneVector s_dot = c * rate;
                    const LaneVector chs_dot = half * c_dot;
                    const LaneVector shs_dot = -(half * c_dot);
                    const LaneVector hs_dot = half * s_dot;
                    for (std::size_t state = 0; state < states; ++state) {
                        const std::size_t base = state * REAL_LANES;
                        const LaneVector p = lane_load(plus + base);
                        const LaneVector m = lane_load(minus + base);
                        const LaneVector z = lane_load(longitudinal + base);
                        const LaneVector dp = lane_load(dot_plus + base);
                        const LaneVector dm = lane_load(dot_minus + base);
                        const LaneVector dz = lane_load(dot_longitudinal + base);
                        lane_store(
                            dot_plus + base,
                            chs * dp + chs_dot * p + shs * dm + shs_dot * m
                                - s * dz - s_dot * z
                        );
                        lane_store(
                            dot_minus + base,
                            shs * dp + shs_dot * p + chs * dm + chs_dot * m
                                + s * dz + s_dot * z
                        );
                        lane_store(
                            dot_longitudinal + base,
                            hs * dp + hs_dot * p - hs * dm - hs_dot * m
                                + c * dz + c_dot * z
                        );
                        lane_store(plus + base, chs * p + shs * m - s * z);
                        lane_store(minus + base, shs * p + chs * m + s * z);
                        lane_store(longitudinal + base, hs * p - hs * m + c * z);
                    }
                }
            } else if (primal.kind[event] == 2 && (action & RECORD) != 0) {
                const std::int64_t offset = primal.output_index[event];
                for (std::int64_t lane = 0; lane < view.active; ++lane) {
                    const std::int64_t index =
                        ((view.train_begin + lane) * primal.atom_count + atom)
                            * output_count
                        + offset;
                    primal.output_real[index] = 0.0F;
                    primal.output_imag[index] =
                        dot_m0 * plus[lane] + m0 * dot_plus[lane];
                }
            }
            if ((action & POST_SHIFT) != 0) {
                shift_real_lanes(plus, minus, states);
                shift_real_lanes(dot_plus, dot_minus, states);
            }
            if ((action & SPOIL_AFTER) != 0) {
                std::fill(plus, plus + width, 0.0F);
                std::fill(minus, minus + width, 0.0F);
                std::fill(dot_plus, dot_plus + width, 0.0F);
                std::fill(dot_minus, dot_minus + width, 0.0F);
            } else if ((action & SHIFT_AFTER) != 0) {
                shift_real_lanes(plus, minus, states);
                shift_real_lanes(dot_plus, dot_minus, states);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Train-axis vectorization.
//
// A block of trains shares one atom, so every per-atom quantity is uniform
// across lanes while flip, phase and duration vary with the train. States are
// stored [state][lane] with the lane axis contiguous and innermost, which is
// what lets the state loops below compile to packed arithmetic. Transcendentals
// stay scalar: they are evaluated once per event, against a state loop that
// runs `state_count` times.
//
// Inactive lanes in a partial block carry a copy of the first train, so the
// arithmetic is uniform; their outputs are simply not written back.
// ---------------------------------------------------------------------------

constexpr std::size_t LANES = 8;

struct LaneStates {
    std::vector<float> storage;
    std::size_t states;
    float* plane[6];

    explicit LaneStates(const std::size_t state_count)
        : storage(6U * state_count * LANES, 0.0F), states(state_count) {
        for (std::size_t index = 0; index < 6U; ++index) {
            plane[index] = storage.data() + index * state_count * LANES;
        }
    }

    void reset() { std::fill(storage.begin(), storage.end(), 0.0F); }

    float* fplus_real() { return plane[0]; }
    float* fplus_imag() { return plane[1]; }
    float* fminus_real() { return plane[2]; }
    float* fminus_imag() { return plane[3]; }
    float* longitudinal_real() { return plane[4]; }
    float* longitudinal_imag() { return plane[5]; }
};

inline void shift_lanes(
    float* fplus_real,
    float* fplus_imag,
    float* fminus_real,
    float* fminus_imag,
    const std::size_t states
) {
    // No states is impossible -- every entry point rejects a state count
    // below one -- but not provable here, and the loops below count down
    // from ``states - 1``.
    if (states == 0) {
        return;
    }
    for (std::size_t state = 0; state + 1 < states; ++state) {
        const std::size_t destination = state * LANES;
        const std::size_t source = (state + 1) * LANES;
        for (std::size_t lane = 0; lane < LANES; ++lane) {
            fminus_real[destination + lane] = fminus_real[source + lane];
            fminus_imag[destination + lane] = fminus_imag[source + lane];
        }
    }
    for (std::size_t lane = 0; lane < LANES; ++lane) {
        fminus_real[(states - 1) * LANES + lane] = 0.0F;
        fminus_imag[(states - 1) * LANES + lane] = 0.0F;
    }
    for (std::size_t state = states - 1; state > 0; --state) {
        const std::size_t destination = state * LANES;
        const std::size_t source = (state - 1) * LANES;
        for (std::size_t lane = 0; lane < LANES; ++lane) {
            fplus_real[destination + lane] = fplus_real[source + lane];
            fplus_imag[destination + lane] = fplus_imag[source + lane];
        }
    }
    // fplus[0] = conj(fminus[0]), reading fminus after its own shift.
    for (std::size_t lane = 0; lane < LANES; ++lane) {
        fplus_real[lane] = fminus_real[lane];
        fplus_imag[lane] = -fminus_imag[lane];
    }
}

// Nine complex coefficients of the RF rotation, one set per lane.
struct RotationLanes {
    float real[9][LANES];
    float imag[9][LANES];
};

inline void build_rotation(
    RotationLanes& rotation,
    const float* alpha,
    const float* phi
) {
    for (std::size_t lane = 0; lane < LANES; ++lane) {
        const float cosine = std::cos(alpha[lane]);
        const float sine = std::sin(alpha[lane]);
        const float cosine_half_sq = 0.5F * (1.0F + cosine);
        const float sine_half_sq = 0.5F * (1.0F - cosine);
        const float phase_real = std::cos(phi[lane]);
        const float phase_imag = std::sin(phi[lane]);
        const float two_real = phase_real * phase_real - phase_imag * phase_imag;
        const float two_imag = 2.0F * phase_real * phase_imag;

        rotation.real[0][lane] = cosine_half_sq;
        rotation.imag[0][lane] = 0.0F;
        rotation.real[1][lane] = sine_half_sq * two_real;
        rotation.imag[1][lane] = sine_half_sq * two_imag;
        // (-i sine) * phase_one
        rotation.real[2][lane] = sine * phase_imag;
        rotation.imag[2][lane] = -sine * phase_real;
        rotation.real[3][lane] = rotation.real[1][lane];
        rotation.imag[3][lane] = -rotation.imag[1][lane];
        rotation.real[4][lane] = cosine_half_sq;
        rotation.imag[4][lane] = 0.0F;
        // (i sine) * conj(phase_one)
        rotation.real[5][lane] = sine * phase_imag;
        rotation.imag[5][lane] = sine * phase_real;
        // (-i sine / 2) * conj(phase_one)
        rotation.real[6][lane] = -0.5F * sine * phase_imag;
        rotation.imag[6][lane] = -0.5F * sine * phase_real;
        // (i sine / 2) * phase_one
        rotation.real[7][lane] = -0.5F * sine * phase_imag;
        rotation.imag[7][lane] = 0.5F * sine * phase_real;
        rotation.real[8][lane] = cosine;
        rotation.imag[8][lane] = 0.0F;
    }
}

inline void rotate_lanes(
    LaneStates& lane_states,
    const RotationLanes& rotation,
    const std::size_t states
) {
    float* fplus_real = lane_states.fplus_real();
    float* fplus_imag = lane_states.fplus_imag();
    float* fminus_real = lane_states.fminus_real();
    float* fminus_imag = lane_states.fminus_imag();
    float* longitudinal_real = lane_states.longitudinal_real();
    float* longitudinal_imag = lane_states.longitudinal_imag();

    for (std::size_t state = 0; state < states; ++state) {
        const std::size_t base = state * LANES;
        for (std::size_t lane = 0; lane < LANES; ++lane) {
            const std::size_t slot = base + lane;
            const float fp_re = fplus_real[slot];
            const float fp_im = fplus_imag[slot];
            const float fm_re = fminus_real[slot];
            const float fm_im = fminus_imag[slot];
            const float z_re = longitudinal_real[slot];
            const float z_im = longitudinal_imag[slot];

            fplus_real[slot] = rotation.real[0][lane] * fp_re
                - rotation.imag[0][lane] * fp_im
                + rotation.real[1][lane] * fm_re - rotation.imag[1][lane] * fm_im
                + rotation.real[2][lane] * z_re - rotation.imag[2][lane] * z_im;
            fplus_imag[slot] = rotation.real[0][lane] * fp_im
                + rotation.imag[0][lane] * fp_re
                + rotation.real[1][lane] * fm_im + rotation.imag[1][lane] * fm_re
                + rotation.real[2][lane] * z_im + rotation.imag[2][lane] * z_re;

            fminus_real[slot] = rotation.real[3][lane] * fp_re
                - rotation.imag[3][lane] * fp_im
                + rotation.real[4][lane] * fm_re - rotation.imag[4][lane] * fm_im
                + rotation.real[5][lane] * z_re - rotation.imag[5][lane] * z_im;
            fminus_imag[slot] = rotation.real[3][lane] * fp_im
                + rotation.imag[3][lane] * fp_re
                + rotation.real[4][lane] * fm_im + rotation.imag[4][lane] * fm_re
                + rotation.real[5][lane] * z_im + rotation.imag[5][lane] * z_re;

            longitudinal_real[slot] = rotation.real[6][lane] * fp_re
                - rotation.imag[6][lane] * fp_im
                + rotation.real[7][lane] * fm_re - rotation.imag[7][lane] * fm_im
                + rotation.real[8][lane] * z_re - rotation.imag[8][lane] * z_im;
            longitudinal_imag[slot] = rotation.real[6][lane] * fp_im
                + rotation.imag[6][lane] * fp_re
                + rotation.real[7][lane] * fm_im + rotation.imag[7][lane] * fm_re
                + rotation.real[8][lane] * z_im + rotation.imag[8][lane] * z_re;
        }
    }
}

// Blocks of trains for one atom; ``work`` indexes the (atom, block) product.
void simulate_lane_range(
    const Buffers& buffers,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count
) {
    const std::size_t states = static_cast<std::size_t>(state_count);
    const std::int64_t blocks_per_atom =
        (buffers.train_count + static_cast<std::int64_t>(LANES) - 1)
        / static_cast<std::int64_t>(LANES);

    LaneStates lane_states(states);
    RotationLanes rotation{};
    alignas(64) float duration[LANES];
    alignas(64) float recovery[LANES];
    alignas(64) float off_real[LANES];
    alignas(64) float off_imag[LANES];
    alignas(64) float alpha[LANES];
    alignas(64) float phi[LANES];
    alignas(64) float demodulation_real[LANES];
    alignas(64) float demodulation_imag[LANES];
    std::int64_t train_of[LANES];

    for (std::int64_t work = work_begin; work < work_end; ++work) {
        const std::int64_t atom = work / blocks_per_atom;
        const std::int64_t block = work % blocks_per_atom;
        const std::int64_t train_begin = block * static_cast<std::int64_t>(LANES);
        const std::int64_t active =
            std::min<std::int64_t>(LANES, buffers.train_count - train_begin);
        for (std::size_t lane = 0; lane < LANES; ++lane) {
            const std::int64_t offset = std::min<std::int64_t>(
                static_cast<std::int64_t>(lane), active - 1
            );
            train_of[lane] = train_begin + offset;
        }

        lane_states.reset();
        float* fplus_real = lane_states.fplus_real();
        float* fplus_imag = lane_states.fplus_imag();
        float* fminus_real = lane_states.fminus_real();
        float* fminus_imag = lane_states.fminus_imag();
        float* longitudinal_real = lane_states.longitudinal_real();
        float* longitudinal_imag = lane_states.longitudinal_imag();
        for (std::size_t lane = 0; lane < LANES; ++lane) {
            longitudinal_real[lane] = 1.0F;
        }

        const float r1 = 1000.0F / buffers.t1[atom];
        const float r2 = 1000.0F / buffers.t2[atom];
        const float b0 = buffers.off_axis ? buffers.b0[atom] : 0.0F;
        const float b1 = buffers.transmit ? buffers.b1[atom] : 1.0F;
        const float b1_phase = buffers.off_axis ? buffers.b1_phase[atom] : 0.0F;
        const float m0 = buffers.density ? buffers.m0[atom] : 1.0F;
        const float efficiency = -scaled(
            1.0F, buffers.inversion_efficiency[atom], buffers.inverting
        );

        for (std::int64_t event = 0; event < event_count; ++event) {
            for (std::size_t lane = 0; lane < LANES; ++lane) {
                duration[lane] =
                    buffers.duration[train_of[lane] * event_count + event];
            }
            // Transcendentals first, once per lane; the state loops that follow
            // keep the lane axis innermost and contiguous so they vectorize.
            for (std::size_t lane = 0; lane < LANES; ++lane) {
                const float dt = duration[lane];
                const float e2 = std::exp(-r2 * dt);
                recovery[lane] = std::exp(-r1 * dt);
                if (buffers.off_axis) {
                    const float angle = -2.0F * PI * b0 * dt;
                    off_real[lane] = e2 * std::cos(angle);
                    off_imag[lane] = e2 * std::sin(angle);
                } else {
                    off_real[lane] = e2;
                    off_imag[lane] = 0.0F;
                }
            }
            for (std::size_t state = 0; state < states; ++state) {
                const std::size_t base = state * LANES;
                for (std::size_t lane = 0; lane < LANES; ++lane) {
                    const std::size_t slot = base + lane;
                    const float fp_re = fplus_real[slot];
                    const float fp_im = fplus_imag[slot];
                    fplus_real[slot] = fp_re * off_real[lane] - fp_im * off_imag[lane];
                    fplus_imag[slot] = fp_re * off_imag[lane] + fp_im * off_real[lane];
                    const float fm_re = fminus_real[slot];
                    const float fm_im = fminus_imag[slot];
                    fminus_real[slot] = fm_re * off_real[lane] + fm_im * off_imag[lane];
                    fminus_imag[slot] = -fm_re * off_imag[lane] + fm_im * off_real[lane];
                    longitudinal_real[slot] *= recovery[lane];
                    longitudinal_imag[slot] *= recovery[lane];
                }
            }
            for (std::size_t lane = 0; lane < LANES; ++lane) {
                longitudinal_real[lane] += 1.0F - recovery[lane];
            }

            const std::uint8_t action = buffers.action[event];
            if ((action & PRE_SHIFT) != 0) {
                shift_lanes(
                    fplus_real, fplus_imag, fminus_real, fminus_imag, states
                );
            }
            if (buffers.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    for (std::size_t slot = 0; slot < states * LANES; ++slot) {
                        longitudinal_real[slot] *= efficiency;
                        longitudinal_imag[slot] *= efficiency;
                    }
                } else {
                    for (std::size_t lane = 0; lane < LANES; ++lane) {
                        const std::int64_t index =
                            train_of[lane] * event_count + event;
                        alpha[lane] = buffers.flip[index] * b1;
                        phi[lane] = buffers.phase[index] + b1_phase;
                    }
                    build_rotation(rotation, alpha, phi);
                    rotate_lanes(lane_states, rotation, states);
                }
            } else if (buffers.kind[event] == 2 && (action & RECORD) != 0) {
                const std::int64_t output = buffers.output_index[event];
                for (std::size_t lane = 0; lane < LANES; ++lane) {
                    const float angle =
                        -buffers.phase[train_of[lane] * event_count + event];
                    demodulation_real[lane] = std::cos(angle);
                    demodulation_imag[lane] = std::sin(angle);
                }
                for (std::int64_t lane = 0; lane < active; ++lane) {
                    const float fp_re = fplus_real[lane];
                    const float fp_im = fplus_imag[lane];
                    const std::int64_t index =
                        ((train_begin + lane) * buffers.atom_count + atom)
                            * output_count
                        + output;
                    buffers.output_real[index] = m0
                        * (fp_re * demodulation_real[lane]
                           - fp_im * demodulation_imag[lane]);
                    buffers.output_imag[index] = m0
                        * (fp_re * demodulation_imag[lane]
                           + fp_im * demodulation_real[lane]);
                }
            }
            if ((action & POST_SHIFT) != 0) {
                shift_lanes(
                    fplus_real, fplus_imag, fminus_real, fminus_imag, states
                );
            }
            if ((action & SPOIL_AFTER) != 0) {
                std::fill(fplus_real, fplus_real + states * LANES, 0.0F);
                std::fill(fplus_imag, fplus_imag + states * LANES, 0.0F);
                std::fill(fminus_real, fminus_real + states * LANES, 0.0F);
                std::fill(fminus_imag, fminus_imag + states * LANES, 0.0F);
            } else if ((action & SHIFT_AFTER) != 0) {
                shift_lanes(
                    fplus_real, fplus_imag, fminus_real, fminus_imag, states
                );
            }
        }
    }
}

template <RfMode MODE, Pools POOLS>
void simulate_range(
    const Buffers& buffers,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count
) {
    // Named rather than passed as flags, so the blocks below read as the
    // physics they are; see ``Pools``. ``bound`` is whichever second pool the
    // longitudinal step pairs the free water with -- the semisolid one when it
    // is the only one, the exchanging one otherwise -- and ``semisolid`` is the
    // third, which only a three-pool run carries.
    constexpr bool MT = POOLS == Pools::SEMISOLID;
    constexpr bool BM = POOLS == Pools::EXCHANGING;
    constexpr bool THREE = POOLS == Pools::THREE;
    constexpr bool TWO_POOL = MT || BM;
    // Which pool carries a transverse pair of its own, and which run absorbs
    // the power a pulse deposits.
    constexpr bool PAIRED = BM || THREE;
    constexpr bool SATURATED = MT || THREE;
    // Hoisted out of every loop below. ``turning`` is whether anything puts a
    // phase on the transverse states at all -- off-resonance and flow reach
    // them through one rotation -- and ``flowing`` is the velocity terms
    // alone, which is what the longitudinal states turn through.
    const bool turning = buffers.off_axis || buffers.moving;
    const bool flowing = buffers.moving;
    const std::size_t states = static_cast<std::size_t>(state_count);
    std::vector<Complex> fplus(states);
    std::vector<Complex> fminus(states);
    std::vector<Complex> longitudinal(states);
    // A second pool always carries longitudinal states. The semisolid one
    // carries nothing else -- it has no transverse magnetization for a
    // gradient to dephase -- while the chemically exchanging one carries a
    // full transverse pair of its own, which shifts and rotates alongside the
    // free water's.
    std::vector<Complex> bound((TWO_POOL || THREE) ? states : 0U);
    std::vector<Complex> semisolid(THREE ? states : 0U);
    std::vector<Complex> bound_plus(PAIRED ? states : 0U);
    std::vector<Complex> bound_minus(PAIRED ? states : 0U);
    Damping<float> damping(states);

    for (std::int64_t work = work_begin; work < work_end; ++work) {
        const TrainView view = train_view(buffers, work, event_count, output_count);
        const std::int64_t atom = view.atom;
        const std::int64_t location = slice_row<MODE>(buffers, atom);
        std::fill(fplus.begin(), fplus.end(), Complex{});
        std::fill(fminus.begin(), fminus.end(), Complex{});
        std::fill(longitudinal.begin(), longitudinal.end(), Complex{});
        const float bound_fraction = (BM || THREE)
            ? buffers.pool_b_fraction[atom]
            : (MT ? buffers.bound_fraction[atom] : 0.0F);
        const float semisolid_fraction =
            SATURATED ? buffers.bound_fraction[atom] : 0.0F;
        longitudinal[0] = Complex(
            1.0F - bound_fraction - (THREE ? semisolid_fraction : 0.0F), 0.0F
        );
        if constexpr (TWO_POOL || THREE) {
            std::fill(bound.begin(), bound.end(), Complex{});
            bound[0] = Complex(bound_fraction, 0.0F);
        }
        if constexpr (THREE) {
            std::fill(semisolid.begin(), semisolid.end(), Complex{});
            semisolid[0] = Complex(semisolid_fraction, 0.0F);
        }
        if constexpr (PAIRED) {
            std::fill(bound_plus.begin(), bound_plus.end(), Complex{});
            std::fill(bound_minus.begin(), bound_minus.end(), Complex{});
        }

        const float r1 = 1000.0F / buffers.t1[atom];
        const float r2 = 1000.0F / buffers.t2[atom];
        const float r1_bound = (BM || THREE)
            ? 1000.0F / buffers.t1_pool_b[atom]
            : (MT ? 1000.0F / buffers.t1_bound[atom] : 0.0F);
        const float r1_semisolid =
            SATURATED ? 1000.0F / buffers.t1_bound[atom] : 0.0F;
        const float r2_bound = PAIRED ? 1000.0F / buffers.t2_pool_b[atom] : 0.0F;
        const float pool_shift = PAIRED ? buffers.pool_b_shift[atom] : 0.0F;
        const float exchange = (BM || THREE)
            ? buffers.pool_b_exchange[atom]
            : (MT ? buffers.bound_exchange[atom] : 0.0F);
        const float semisolid_exchange =
            SATURATED ? buffers.bound_exchange[atom] : 0.0F;
        // A semisolid pool holds a share of the voxel without carrying any
        // transverse magnetization, so it is absent from the 2x2 below and
        // present in how much free water that 2x2's exchange sees.
        const float transverse_free =
            1.0F - bound_fraction - (THREE ? semisolid_fraction : 0.0F);
        const float damping_rate =
            buffers.diffusing ? buffers.diffusion[atom] : 0.0F;
        const float velocity = flowing ? buffers.velocity[atom] : 0.0F;
        const float flow_rate = velocity * buffers.flow_scale;
        const float washout_rate =
            std::fabs(velocity) * buffers.washout_scale;
        // With one shim the transmit field is a property of the voxel and
        // lifts out of the event loop; with several it belongs to the shim a
        // pulse drives, and is read where the pulse is.
        const float b1 = buffers.transmit ? buffers.b1[atom] : 1.0F;
        const float b1_phase = buffers.off_axis ? buffers.b1_phase[atom] : 0.0F;
        for (std::int64_t event = 0; event < event_count; ++event) {
            const float dt = view.duration[event];
            damping.set(damping_rate, dt);
            const float wout = flowing ? washout_out(washout_rate, dt) : 1.0F;
            const float e1 = std::exp(-r1 * dt) * wout;
            const float e2 = std::exp(-r2 * dt) * wout;
            // The exchange operator is a property of the interval, not of a
            // dephasing order, so it is formed once here; the per-order
            // diffusion and flow damping multiply it below.
            const TwoPoolStep<float> pools = TWO_POOL
                ? two_pool_step(r1, r1_bound, exchange, bound_fraction, dt, wout)
                : TwoPoolStep<float>{};
            // The transverse pair is a property of the interval on the same
            // terms, so it is formed once too; a single-pool run has no second
            // transverse state for it to act on.
            // Three pools carry the same 2x2 transverse operator as two: only
            // the free water and the exchanging pool have transverse states,
            // and the semisolid pool reaches the answer along Z alone.
            const ThreePoolStep<double> triple = THREE
                ? three_pool_step<double>(
                    r1, r1_bound, r1_semisolid, exchange, semisolid_exchange,
                    bound_fraction, semisolid_fraction, dt, wout
                )
                : ThreePoolStep<double>{};
            const TwoPoolTransverse<Complex> across = PAIRED
                ? two_pool_transverse_step<float, Complex>(
                    r2, r2_bound, exchange, bound_fraction, transverse_free, pool_shift, dt,
                    wout
                )
                : TwoPoolTransverse<Complex>{};
            const float off_angle =
                buffers.off_axis ? -2.0F * PI * buffers.b0[atom] * dt : 0.0F;
            for (std::int64_t state = 0; state < state_count; ++state) {
                const std::size_t index = static_cast<std::size_t>(state);
                const float damp_transverse = damping.transverse[index];
                float turn_longitudinal = 0.0F;
                float turn_transverse = 0.0F;
                if (flowing) {
                    flow_turn(
                        flow_rate, dt, index, turn_longitudinal, turn_transverse
                    );
                }
                // Flow winds the transverse states through the same rotation
                // off-resonance does, so the two phases add before either is
                // taken; the longitudinal states carry a phase of their own.
                if constexpr (PAIRED) {
                    // Both pools take the same off-resonance and the same
                    // per-order damping; what separates them is the chemical
                    // shift, which the exchange operator already carries.
                    const Complex carried = turned(
                        damp_transverse, off_angle + turn_transverse, turning
                    );
                    const Complex free_plus = fplus[index];
                    const Complex bound_plus_state = bound_plus[index];
                    const Complex free_minus = fminus[index];
                    const Complex bound_minus_state = bound_minus[index];
                    fplus[index] =
                        (across.e11 * free_plus + across.e12 * bound_plus_state)
                        * carried;
                    bound_plus[index] =
                        (across.e21 * free_plus + across.e22 * bound_plus_state)
                        * carried;
                    // ``F-`` takes the conjugate of the operator entry by
                    // entry, not its transpose: it is the conjugate state, and
                    // the map it follows is the conjugate map.
                    const Complex conjugated = std::conj(carried);
                    fminus[index] = (std::conj(across.e11) * free_minus
                        + std::conj(across.e12) * bound_minus_state) * conjugated;
                    bound_minus[index] = (std::conj(across.e21) * free_minus
                        + std::conj(across.e22) * bound_minus_state) * conjugated;
                } else {
                    const Complex transverse = turned(
                        e2 * damp_transverse, off_angle + turn_transverse, turning
                    );
                    fplus[index] *= transverse;
                    fminus[index] *= std::conj(transverse);
                }
                // Both pools take the same per-order damping and phase: their
                // order-n states describe one dephasing configuration, and a
                // second pool has no diffusion coefficient of its own.
                const Complex spin =
                    turned(damping.longitudinal[index], turn_longitudinal, flowing);
                if constexpr (THREE) {
                    // The operator was formed in double and is read here as
                    // the float32 it has been narrowed to; the state loop is
                    // the same arithmetic as every other pool count's.
                    const Complex pools_in[3] = {
                        longitudinal[index], bound[index], semisolid[index]
                    };
                    Complex mixed[3];
                    for (int row = 0; row < 3; ++row) {
                        Complex carried{};
                        for (int column = 0; column < 3; ++column) {
                            carried += static_cast<float>(triple.entry[row][column])
                                * pools_in[column];
                        }
                        mixed[row] = carried * spin;
                    }
                    longitudinal[index] = mixed[0];
                    bound[index] = mixed[1];
                    semisolid[index] = mixed[2];
                } else if constexpr (TWO_POOL) {
                    const Complex free_state = longitudinal[index];
                    const Complex bound_state = bound[index];
                    longitudinal[index] =
                        (pools.e11 * free_state + pools.e12 * bound_state) * spin;
                    bound[index] =
                        (pools.e21 * free_state + pools.e22 * bound_state) * spin;
                } else {
                    longitudinal[index] *= e1 * spin;
                }
            }
            if constexpr (THREE) {
                longitudinal[0] += Complex(static_cast<float>(triple.recovery[0]), 0.0F);
                bound[0] += Complex(static_cast<float>(triple.recovery[1]), 0.0F);
                semisolid[0] += Complex(static_cast<float>(triple.recovery[2]), 0.0F);
            } else if constexpr (TWO_POOL) {
                longitudinal[0] += Complex(pools.recovery_free, 0.0F);
                bound[0] += Complex(pools.recovery_bound, 0.0F);
            } else {
                longitudinal[0] += Complex(1.0F - e1, 0.0F);
            }

            const std::uint8_t action = buffers.action[event];
            if ((action & PRE_SHIFT) != 0) {
                shift(fplus, fminus);
                if constexpr (PAIRED) {
                    shift(bound_plus, bound_minus);
                }
            }
            if (buffers.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    // A semisolid pool's T2 is short enough that an adiabatic
                    // sweep saturates it rather than turning it over, and what
                    // it saturates by is already carried by the pulse's own
                    // saturation term. A chemically exchanging pool is free
                    // water and inverts like any other.
                    const float efficiency = -scaled(
                        1.0F, buffers.inversion_efficiency[atom], buffers.inverting
                    );
                    for (Complex& value : longitudinal) {
                        value *= efficiency;
                    }
                    if constexpr (PAIRED) {
                        for (Complex& value : bound) {
                            value *= efficiency;
                        }
                    }
                } else {
                    const bool shimmed = buffers.shim_count > 1;
                    const std::int64_t transmit =
                        shimmed ? transmit_row(buffers, event, atom) : atom;
                    const float theta = scaled(
                        view.flip[event],
                        shimmed ? buffers.b1[transmit] : b1,
                        buffers.transmit
                    );
                    const float phi = view.phase[event]
                        + (shimmed ? buffers.b1_phase[transmit] : b1_phase);
                    if constexpr (SATURATED) {
                        // The semisolid pool absorbs the power the pulse
                        // deposits, so it reads the bare flip the transmit
                        // field gives the voxel -- not the slice-shaped
                        // rotation the free pool takes from the table.
                        const float offset =
                            buffers.rf_frequency[event] - buffers.b0[atom];
                        const float absorbed = std::exp(
                            buffers.saturation[event] * theta * theta
                            * lineshape_at(buffers, offset)
                        );
                        for (Complex& value : (THREE ? semisolid : bound)) {
                            value *= absorbed;
                        }
                    }
                    if constexpr (MODE != RfMode::INSTANT) {
                        Complex a{};
                        Complex b{};
                        if constexpr (MODE == RfMode::DYNAMIC) {
                            // Already integrated at this pulse's own flip, so
                            // the flip is inside the pair rather than read
                            // against it.
                            dynamic_pair_at(
                                buffers,
                                dynamic_row(buffers, view, event),
                                atom, a, b
                            );
                        } else {
                            profile_pair(
                                buffers,
                                table_row<MODE>(buffers, event, location),
                                theta, a, b
                            );
                        }
                        // Either pair is built at zero RF phase, which turns
                        // the rotation axis and so reaches ``b`` alone.
                        const Complex spun = b * std::polar(1.0F, -phi);
                        rotate_spinor(fplus, fminus, longitudinal, a, spun);
                        if constexpr (PAIRED) {
                            // The same pulse, the same rotation. A chemical
                            // shift moves where a pool precesses, not what a
                            // pulse does to it.
                            rotate_spinor(
                                bound_plus, bound_minus, bound, a, spun
                            );
                        }
                    } else {
                        rotate(fplus, fminus, longitudinal, theta, phi);
                        if constexpr (PAIRED) {
                            rotate(bound_plus, bound_minus, bound, theta, phi);
                        }
                    }
                }
            } else if (buffers.kind[event] == 2 && (action & RECORD) != 0) {
                const std::int64_t output = buffers.output_index[event];
                const Complex demodulation = std::polar(1.0F, -view.phase[event]);
                // A coil sees the whole voxel, so what it records is the sum
                // over pools. Each pool's share of the magnetization is
                // already in its own state, the fractions having split the
                // equilibrium at t = 0.
                const Complex recorded =
                    PAIRED ? fplus[0] + bound_plus[0] : fplus[0];
                const Complex signal = buffers.density
                    ? buffers.m0[atom] * recorded * demodulation
                    : recorded * demodulation;
                const std::int64_t index = view.output_base + output;
                buffers.output_real[index] = signal.real();
                buffers.output_imag[index] = signal.imag();
            }
            if ((action & POST_SHIFT) != 0) {
                shift(fplus, fminus);
                if constexpr (PAIRED) {
                    shift(bound_plus, bound_minus);
                }
            }
            if ((action & SPOIL_AFTER) != 0) {
                std::fill(fplus.begin(), fplus.end(), Complex{});
                std::fill(fminus.begin(), fminus.end(), Complex{});
                if constexpr (PAIRED) {
                    std::fill(bound_plus.begin(), bound_plus.end(), Complex{});
                    std::fill(bound_minus.begin(), bound_minus.end(), Complex{});
                }
            } else if ((action & SHIFT_AFTER) != 0) {
                shift(fplus, fminus);
                if constexpr (PAIRED) {
                    shift(bound_plus, bound_minus);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reverse mode.
//
// The forward pass is a chain of linear maps on the state s = (F+, F-, Z),
// with an affine term from longitudinal recovery. Writing s_k = A_k s_{k-1} +
// b_k, the adjoint recursion is lambda_{k-1} = A_k^H lambda_k, and a real
// parameter theta collects
//
//     dL/dtheta += Re( conj(lambda_k) . (dA_k/dtheta) s_{k-1} + ... ).
//
// For a real loss, Torch hands back grad_output = dL/dRe(y) + i dL/dIm(y),
// which is exactly the adjoint convention used here: the adjoint of v = c u is
// u_bar = conj(c) v_bar, and the adjoint of conjugation is conjugation.
// ---------------------------------------------------------------------------

struct VjpBuffers {
    Buffers primal;
    const float* grad_output_real;
    const float* grad_output_imag;
    float* grad_t1;
    float* grad_t2;
    float* grad_m0;
    float* grad_b1;
    float* grad_b1_phase;
    float* grad_b0;
    float* grad_inversion_efficiency;
    float* grad_diffusion;
    float* grad_velocity;
    // The bound pool's three properties. A single-pool run leaves them at
    // zero, which is the true gradient: the kernel it selects does not take
    // them as arguments.
    float* grad_bound_fraction;
    float* grad_exchange_rate;
    float* grad_t1_bound;
    // The chemically exchanging pool's five. Read only by the kernels that
    // carry it; a run declaring the semisolid pool instead leaves them alone.
    float* grad_pool_b_fraction;
    float* grad_pool_b_exchange;
    float* grad_t1_pool_b;
    float* grad_t2_pool_b;
    float* grad_pool_b_shift;
    // Per-event gradients are shared by every atom. Workers accumulate into
    // private buffers which are reduced in a fixed order, so the result does
    // not depend on thread scheduling.
    float* grad_flip;
    float* grad_phase;
    float* grad_duration;
    // The cotangent on the per-voxel rotations, laid out as they are. A work
    // item is one (train, atom), and a row belongs to one train, so each of
    // these entries has exactly one writer.
    float* grad_dynamic;
    // The cotangent on a tabulated run's table, one table per (train, atom)
    // in work order, so that each entry has one writer too.
    float* grad_pool;
};

using State = std::vector<Complex>;

inline void shift_adjoint(State& fplus_bar, State& fminus_bar) {
    const std::size_t count = fplus_bar.size();
    // An empty state plane is impossible -- every entry point rejects a state
    // count below one -- but not provable here, and the loops below count down
    // from ``count - 1``.
    if (count == 0) {
        return;
    }
    // fplus[0] = conj(fminus[1]) couples the two branches; capture it before
    // the in-place shifts overwrite the source.
    const Complex carry = std::conj(fplus_bar[0]);
    for (std::size_t state = 0; state + 1 < count; ++state) {
        fplus_bar[state] = fplus_bar[state + 1];
    }
    fplus_bar[count - 1] = Complex{};
    for (std::size_t state = count - 1; state > 0; --state) {
        fminus_bar[state] = fminus_bar[state - 1];
    }
    fminus_bar[0] = Complex{};
    if (count > 1) {
        fminus_bar[1] += carry;
    }
}

inline void rotate_adjoint(
    const State& fplus_in,
    const State& fminus_in,
    const State& longitudinal_in,
    State& fplus_bar,
    State& fminus_bar,
    State& longitudinal_bar,
    const float alpha,
    const float phi,
    float& grad_alpha,
    float& grad_phi
) {
    const float cosine = std::cos(alpha);
    const float sine = std::sin(alpha);
    const Complex phase_one = std::polar(1.0F, phi);
    const Complex phase_two = phase_one * phase_one;
    const Complex phase_conj = std::conj(phase_one);

    const Complex t00(0.5F * (1.0F + cosine), 0.0F);
    const Complex t01 = 0.5F * (1.0F - cosine) * phase_two;
    const Complex t02 = Complex(0.0F, -sine) * phase_one;
    const Complex t10 = std::conj(t01);
    const Complex t11 = t00;
    const Complex t12 = Complex(0.0F, sine) * phase_conj;
    const Complex t20 = Complex(0.0F, -0.5F * sine) * phase_conj;
    const Complex t21 = Complex(0.0F, 0.5F * sine) * phase_one;
    const Complex t22(cosine, 0.0F);

    // d/dalpha of each coefficient.
    const Complex d00(-0.5F * sine, 0.0F);
    const Complex d01 = 0.5F * sine * phase_two;
    const Complex d02 = Complex(0.0F, -cosine) * phase_one;
    const Complex d10 = std::conj(d01);
    const Complex d11 = d00;
    const Complex d12 = Complex(0.0F, cosine) * phase_conj;
    const Complex d20 = Complex(0.0F, -0.5F * cosine) * phase_conj;
    const Complex d21 = Complex(0.0F, 0.5F * cosine) * phase_one;
    const Complex d22(-sine, 0.0F);

    // Every coefficient is of the form c * exp(i n phi), so d/dphi = i n t.
    const Complex imaginary(0.0F, 1.0F);
    float alpha_sum = 0.0F;
    float phi_sum = 0.0F;
    for (std::size_t state = 0; state < fplus_bar.size(); ++state) {
        const Complex a0 = fplus_bar[state];
        const Complex a1 = fminus_bar[state];
        const Complex a2 = longitudinal_bar[state];
        const Complex x0 = fplus_in[state];
        const Complex x1 = fminus_in[state];
        const Complex x2 = longitudinal_in[state];

        alpha_sum += std::real(
            std::conj(a0) * (d00 * x0 + d01 * x1 + d02 * x2)
            + std::conj(a1) * (d10 * x0 + d11 * x1 + d12 * x2)
            + std::conj(a2) * (d20 * x0 + d21 * x1 + d22 * x2)
        );
        phi_sum += std::real(
            std::conj(a0) * imaginary * (2.0F * t01 * x1 + t02 * x2)
            + std::conj(a1) * imaginary * (-2.0F * t10 * x0 - t12 * x2)
            + std::conj(a2) * imaginary * (-t20 * x0 + t21 * x1)
        );

        fplus_bar[state] =
            std::conj(t00) * a0 + std::conj(t10) * a1 + std::conj(t20) * a2;
        fminus_bar[state] =
            std::conj(t01) * a0 + std::conj(t11) * a1 + std::conj(t21) * a2;
        longitudinal_bar[state] =
            std::conj(t02) * a0 + std::conj(t12) * a1 + std::conj(t22) * a2;
    }
    grad_alpha += alpha_sum;
    grad_phi += phi_sum;
}

// The adjoint of the spinor rotation: the states back through the conjugate
// transpose, and the pair reached in closed form.
//
// Every entry of the matrix is a product of two factors drawn from the pair and
// its conjugate, so both Wirtinger halves of the pair's cotangent are linear in
// the outer product of the seed with the state the rotation acted on. That
// outer product is accumulated over the states first, so the pair is worked out
// once per pulse rather than once per state.
inline void rotate_adjoint_spinor(
    const State& fplus_in,
    const State& fminus_in,
    const State& longitudinal_in,
    State& fplus_bar,
    State& fminus_bar,
    State& longitudinal_bar,
    const Complex a,
    const Complex b,
    Complex& grad_a,
    Complex& grad_b
) {
    const Complex conj_a = std::conj(a);
    const Complex conj_b = std::conj(b);
    const Complex t00 = conj_a * conj_a;
    const Complex t01 = -conj_b * conj_b;
    const Complex t02 = -2.0F * std::conj(a * b);
    const Complex t10 = -b * b;
    const Complex t11 = a * a;
    const Complex t12 = -2.0F * a * b;
    const Complex t20 = conj_a * b;
    const Complex t21 = a * conj_b;
    const Complex t22(std::norm(a) - std::norm(b), 0.0F);

    Complex m[3][3]{};
    for (std::size_t state = 0; state < fplus_bar.size(); ++state) {
        const Complex a0 = fplus_bar[state];
        const Complex a1 = fminus_bar[state];
        const Complex a2 = longitudinal_bar[state];
        const Complex x0 = fplus_in[state];
        const Complex x1 = fminus_in[state];
        const Complex x2 = longitudinal_in[state];

        const Complex s0 = std::conj(a0);
        const Complex s1 = std::conj(a1);
        const Complex s2 = std::conj(a2);
        m[0][0] += s0 * x0;
        m[0][1] += s0 * x1;
        m[0][2] += s0 * x2;
        m[1][0] += s1 * x0;
        m[1][1] += s1 * x1;
        m[1][2] += s1 * x2;
        m[2][0] += s2 * x0;
        m[2][1] += s2 * x1;
        m[2][2] += s2 * x2;

        fplus_bar[state] =
            std::conj(t00) * a0 + std::conj(t10) * a1 + std::conj(t20) * a2;
        fminus_bar[state] =
            std::conj(t01) * a0 + std::conj(t11) * a1 + std::conj(t21) * a2;
        longitudinal_bar[state] =
            std::conj(t02) * a0 + std::conj(t12) * a1 + std::conj(t22) * a2;
    }

    const Complex holding_conj_a = 2.0F * a * m[1][1] - 2.0F * b * m[1][2]
        + conj_b * m[2][1] + conj_a * m[2][2];
    const Complex holding_a = 2.0F * conj_a * m[0][0] - 2.0F * conj_b * m[0][2]
        + b * m[2][0] + a * m[2][2];
    const Complex holding_conj_b = -2.0F * b * m[1][0] - 2.0F * a * m[1][2]
        + conj_a * m[2][0] - conj_b * m[2][2];
    const Complex holding_b = -2.0F * conj_b * m[0][1] - 2.0F * conj_a * m[0][2]
        + a * m[2][1] - b * m[2][2];
    grad_a += std::conj(holding_conj_a) + holding_a;
    grad_b += std::conj(holding_conj_b) + holding_b;
}

// ---------------------------------------------------------------------------
// Forward-over-reverse.
//
// The JVP computes ydot = J(theta) thetadot. Seeding its adjoint with sigma,
//
//     dL/dthetadot_i = Re<sigma, J_i(theta)>              (the first-order VJP)
//     dL/dtheta_i    = sum_j thetadot_j dV_i/dtheta_j     (its directional
//                                                          derivative)
//
// where V_i is that same first-order VJP. So running the adjoint above on dual
// numbers seeded with thetadot yields both halves at once: the value part is
// the tangent gradient, the tangent part is the primal gradient. No second
// derivative has to be written out by hand; dual arithmetic produces it.
// ---------------------------------------------------------------------------

using DualState = std::vector<DualComplex>;

inline void shift_adjoint(DualState& fplus_bar, DualState& fminus_bar) {
    const std::size_t count = fplus_bar.size();
    // An empty state plane is impossible -- every entry point rejects a state
    // count below one -- but not provable here, and the loops below count down
    // from ``count - 1``.
    if (count == 0) {
        return;
    }
    const DualComplex carry = conjugate(fplus_bar[0]);
    for (std::size_t state = 0; state + 1 < count; ++state) {
        fplus_bar[state] = fplus_bar[state + 1];
    }
    fplus_bar[count - 1] = DualComplex{};
    for (std::size_t state = count - 1; state > 0; --state) {
        fminus_bar[state] = fminus_bar[state - 1];
    }
    fminus_bar[0] = DualComplex{};
    if (count > 1) {
        fminus_bar[1] = fminus_bar[1] + carry;
    }
}

// Inlined into the vector clones below rather than called from them: a
// clone is a copy of the caller, so a body reached through a call is
// compiled once for the baseline instruction set and no more.
template <RfMode MODE, Pools POOLS>
TORCHSIM_ALWAYS_INLINE void simulate_vjp_range(
    const VjpBuffers& buffers,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count,
    float* grad_flip_local,
    float* grad_phase_local,
    float* grad_duration_local,
    // Per-atom accumulators laid out [row][atom], the rows given by
    // TissueLayout. Work items are split across the (atom, train) product, so
    // several threads reach the same atom and every train contributes to it.
    float* grad_tissue_local
) {
    constexpr bool MT = POOLS == Pools::SEMISOLID;
    constexpr bool BM = POOLS == Pools::EXCHANGING;
    constexpr bool THREE = POOLS == Pools::THREE;
    constexpr bool TWO_POOL = MT || BM;
    constexpr bool PAIRED = BM || THREE;
    constexpr bool SATURATED = MT || THREE;
    const Buffers& primal = buffers.primal;
    const TissueLayout layout(primal.shim_count);
    const std::int64_t atoms = primal.atom_count;
    // Hoisted out of the three state loops below -- the recording forward,
    // the replay and the adjoint -- each of which takes two turns per state
    // per event.
    const bool turning = primal.off_axis || primal.moving;
    const bool flowing = primal.moving;
    const std::size_t states = static_cast<std::size_t>(state_count);
    // A second pool rides along the trajectory as blocks of its own: it enters
    // an event as its own vector and the RF operator acts on it, so the
    // reverse sweep cannot replay it from the free pool's. The semisolid pool
    // adds one block, the chemically exchanging one three.
    const std::size_t trajectory_stride =
        (THREE ? 7U : (BM ? 6U : (MT ? 4U : 3U))) * states;

    // State at the start of every event; intra-event intermediates are cheap
    // enough to replay from it during the reverse sweep.
    std::vector<Complex> trajectory(
        static_cast<std::size_t>(event_count) * trajectory_stride
    );
    State fplus(states);
    State fminus(states);
    State longitudinal(states);
    State fplus_bar(states);
    State fminus_bar(states);
    State longitudinal_bar(states);
    State fplus_relaxed(states);
    State fminus_relaxed(states);
    State longitudinal_relaxed(states);
    State fplus_shifted(states);
    State fminus_shifted(states);
    State longitudinal_pre(states);
    State bound((TWO_POOL || THREE) ? states : 0U);
    State bound_bar((TWO_POOL || THREE) ? states : 0U);
    State bound_relaxed((TWO_POOL || THREE) ? states : 0U);
    State semisolid(THREE ? states : 0U);
    State semisolid_bar(THREE ? states : 0U);
    State semisolid_relaxed(THREE ? states : 0U);
    State bound_plus(PAIRED ? states : 0U);
    State bound_minus(PAIRED ? states : 0U);
    State bound_plus_bar(PAIRED ? states : 0U);
    State bound_minus_bar(PAIRED ? states : 0U);
    State bound_plus_relaxed(PAIRED ? states : 0U);
    State bound_minus_relaxed(PAIRED ? states : 0U);
    State bound_plus_shifted(PAIRED ? states : 0U);
    State bound_minus_shifted(PAIRED ? states : 0U);
    Damping<float> damping(states);

    for (std::int64_t work = work_begin; work < work_end; ++work) {
        const TrainView view = train_view(primal, work, event_count, output_count);
        const std::int64_t atom = view.atom;
        const std::int64_t location = slice_row<MODE>(primal, atom);
        float* const grad_flip_train = grad_flip_local + view.event_base;
        float* const grad_phase_train = grad_phase_local + view.event_base;
        float* const grad_duration_train = grad_duration_local + view.event_base;
        std::fill(fplus.begin(), fplus.end(), Complex{});
        std::fill(fminus.begin(), fminus.end(), Complex{});
        std::fill(longitudinal.begin(), longitudinal.end(), Complex{});
        const float bound_fraction = (BM || THREE)
            ? primal.pool_b_fraction[atom]
            : (MT ? primal.bound_fraction[atom] : 0.0F);
        const float semisolid_fraction =
            SATURATED ? primal.bound_fraction[atom] : 0.0F;
        longitudinal[0] = Complex(
            1.0F - bound_fraction - (THREE ? semisolid_fraction : 0.0F), 0.0F
        );
        if constexpr (THREE) {
            std::fill(semisolid.begin(), semisolid.end(), Complex{});
            semisolid[0] = Complex(semisolid_fraction, 0.0F);
        }
        if constexpr (TWO_POOL || THREE) {
            std::fill(bound.begin(), bound.end(), Complex{});
            bound[0] = Complex(bound_fraction, 0.0F);
        }
        if constexpr (PAIRED) {
            std::fill(bound_plus.begin(), bound_plus.end(), Complex{});
            std::fill(bound_minus.begin(), bound_minus.end(), Complex{});
        }

        const float t1 = primal.t1[atom];
        const float t2 = primal.t2[atom];
        const float r1 = 1000.0F / t1;
        const float r2 = 1000.0F / t2;
        const float t1_bound = (BM || THREE)
            ? primal.t1_pool_b[atom]
            : (MT ? primal.t1_bound[atom] : 0.0F);
        const float r1_bound = (TWO_POOL || THREE) ? 1000.0F / t1_bound : 0.0F;
        const float t1_semisolid = SATURATED ? primal.t1_bound[atom] : 0.0F;
        const float r1_semisolid = SATURATED ? 1000.0F / t1_semisolid : 0.0F;
        const float t2_bound = PAIRED ? primal.t2_pool_b[atom] : 0.0F;
        const float r2_bound = PAIRED ? 1000.0F / t2_bound : 0.0F;
        const float pool_shift = PAIRED ? primal.pool_b_shift[atom] : 0.0F;
        const float exchange = (BM || THREE)
            ? primal.pool_b_exchange[atom]
            : (MT ? primal.bound_exchange[atom] : 0.0F);
        const float semisolid_exchange =
            SATURATED ? primal.bound_exchange[atom] : 0.0F;
        const float transverse_free =
            1.0F - bound_fraction - (THREE ? semisolid_fraction : 0.0F);
        const float b0 = primal.off_axis ? primal.b0[atom] : 0.0F;
        const float m0 = primal.density ? primal.m0[atom] : 1.0F;
        // With one shim the transmit field is a property of the voxel and
        // lifts out of the event loop; with several it belongs to the shim a
        // pulse drives, and is read where the pulse is.
        const float b1 = primal.transmit ? primal.b1[atom] : 1.0F;
        const float b1_phase = primal.off_axis ? primal.b1_phase[atom] : 0.0F;
        const bool shimmed = primal.shim_count > 1;
        const float efficiency =
            primal.inverting ? primal.inversion_efficiency[atom] : 1.0F;
        const float damping_rate =
            primal.diffusing ? primal.diffusion[atom] : 0.0F;
        const float velocity = flowing ? primal.velocity[atom] : 0.0F;
        const float flow_rate = velocity * primal.flow_scale;
        const float washout_rate = std::fabs(velocity) * primal.washout_scale;

        // ---- forward, recording the state entering each event ----
        for (std::int64_t event = 0; event < event_count; ++event) {
            Complex* slot = trajectory.data()
                + static_cast<std::size_t>(event) * trajectory_stride;
            std::copy(fplus.begin(), fplus.end(), slot);
            std::copy(fminus.begin(), fminus.end(), slot + states);
            std::copy(longitudinal.begin(), longitudinal.end(), slot + 2U * states);
            if constexpr (TWO_POOL || THREE) {
                std::copy(bound.begin(), bound.end(), slot + 3U * states);
            }
            if constexpr (PAIRED) {
                std::copy(bound_plus.begin(), bound_plus.end(), slot + 4U * states);
                std::copy(
                    bound_minus.begin(), bound_minus.end(), slot + 5U * states
                );
            }
            if constexpr (THREE) {
                std::copy(semisolid.begin(), semisolid.end(), slot + 6U * states);
            }

            const float dt = view.duration[event];
            damping.set(damping_rate, dt);
            const float wout = washout_out(washout_rate, dt);
            const float e1 = std::exp(-r1 * dt) * wout;
            const float e2 = std::exp(-r2 * dt) * wout;
            const TwoPoolStep<float> pools = TWO_POOL
                ? two_pool_step(r1, r1_bound, exchange, bound_fraction, dt, wout)
                : TwoPoolStep<float>{};
            const ThreePoolStep<double> triple = THREE
                ? three_pool_step<double>(
                    r1, r1_bound, r1_semisolid, exchange, semisolid_exchange,
                    bound_fraction, semisolid_fraction, dt, wout
                )
                : ThreePoolStep<double>{};
            const TwoPoolTransverse<Complex> across = PAIRED
                ? two_pool_transverse_step<float, Complex>(
                    r2, r2_bound, exchange, bound_fraction, transverse_free, pool_shift, dt, wout
                )
                : TwoPoolTransverse<Complex>{};
            const float off_angle = -2.0F * PI * b0 * dt;
            for (std::size_t state = 0; state < states; ++state) {
                float turn_longitudinal = 0.0F;
                float turn_transverse = 0.0F;
                if (flowing) {
                    flow_turn(
                        flow_rate, dt, state, turn_longitudinal, turn_transverse
                    );
                }
                if constexpr (PAIRED) {
                    const Complex carried = turned(
                        damping.transverse[state], off_angle + turn_transverse,
                        turning
                    );
                    const Complex free_plus = fplus[state];
                    const Complex pool_plus = bound_plus[state];
                    const Complex free_minus = fminus[state];
                    const Complex pool_minus = bound_minus[state];
                    fplus[state] =
                        (across.e11 * free_plus + across.e12 * pool_plus) * carried;
                    bound_plus[state] =
                        (across.e21 * free_plus + across.e22 * pool_plus) * carried;
                    const Complex conjugated = std::conj(carried);
                    fminus[state] = (std::conj(across.e11) * free_minus
                        + std::conj(across.e12) * pool_minus) * conjugated;
                    bound_minus[state] = (std::conj(across.e21) * free_minus
                        + std::conj(across.e22) * pool_minus) * conjugated;
                } else {
                    const Complex transverse = turned(
                        e2 * damping.transverse[state],
                        off_angle + turn_transverse, turning
                    );
                    fplus[state] *= transverse;
                    fminus[state] *= std::conj(transverse);
                }
                const Complex spin = turned(
                    damping.longitudinal[state], turn_longitudinal, flowing
                );
                if constexpr (THREE) {
                    const Complex pools_in[3] = {
                        longitudinal[state], bound[state], semisolid[state]
                    };
                    Complex mixed[3];
                    for (int row = 0; row < 3; ++row) {
                        Complex carried{};
                        for (int column = 0; column < 3; ++column) {
                            carried += static_cast<float>(triple.entry[row][column])
                                * pools_in[column];
                        }
                        mixed[row] = carried * spin;
                    }
                    longitudinal[state] = mixed[0];
                    bound[state] = mixed[1];
                    semisolid[state] = mixed[2];
                } else if constexpr (TWO_POOL) {
                    const Complex free_state = longitudinal[state];
                    const Complex bound_state = bound[state];
                    longitudinal[state] =
                        (pools.e11 * free_state + pools.e12 * bound_state) * spin;
                    bound[state] =
                        (pools.e21 * free_state + pools.e22 * bound_state) * spin;
                } else {
                    longitudinal[state] *= e1 * spin;
                }
            }
            if constexpr (THREE) {
                longitudinal[0] += Complex(static_cast<float>(triple.recovery[0]), 0.0F);
                bound[0] += Complex(static_cast<float>(triple.recovery[1]), 0.0F);
                semisolid[0] += Complex(static_cast<float>(triple.recovery[2]), 0.0F);
            } else if constexpr (TWO_POOL) {
                longitudinal[0] += Complex(pools.recovery_free, 0.0F);
                bound[0] += Complex(pools.recovery_bound, 0.0F);
            } else {
                longitudinal[0] += Complex(1.0F - e1, 0.0F);
            }

            const std::uint8_t action = primal.action[event];
            if ((action & PRE_SHIFT) != 0) {
                shift(fplus, fminus);
                if constexpr (PAIRED) {
                    shift(bound_plus, bound_minus);
                }
            }
            if (primal.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    for (Complex& value : longitudinal) {
                        value *= -efficiency;
                    }
                    if constexpr (PAIRED) {
                        for (Complex& value : bound) {
                            value *= -efficiency;
                        }
                    }
                } else {
                    const std::int64_t transmit =
                        shimmed ? transmit_row(primal, event, atom) : atom;
                    const float alpha = view.flip[event]
                        * (shimmed ? primal.b1[transmit] : b1);
                    const float phi = view.phase[event]
                        + (shimmed ? primal.b1_phase[transmit] : b1_phase);
                    if constexpr (SATURATED) {
                        const float absorbed = std::exp(
                            primal.saturation[event] * alpha * alpha
                            * lineshape_at(
                                primal, primal.rf_frequency[event] - b0
                            )
                        );
                        for (Complex& value : (THREE ? semisolid : bound)) {
                            value *= absorbed;
                        }
                    }
                    if constexpr (MODE != RfMode::INSTANT) {
                        Complex pair_a{};
                        Complex pair_b{};
                        if constexpr (MODE == RfMode::DYNAMIC) {
                            dynamic_pair_at(
                                primal,
                                dynamic_row(primal, view, event),
                                atom, pair_a, pair_b
                            );
                        } else {
                            profile_pair(
                                primal,
                                table_row<MODE>(primal, event, location),
                                alpha, pair_a, pair_b
                            );
                        }
                        const Complex spun = pair_b * std::polar(1.0F, -phi);
                        rotate_spinor(
                            fplus, fminus, longitudinal, pair_a, spun
                        );
                        if constexpr (PAIRED) {
                            rotate_spinor(
                                bound_plus, bound_minus, bound, pair_a, spun
                            );
                        }
                    } else {
                        rotate(fplus, fminus, longitudinal, alpha, phi);
                        if constexpr (PAIRED) {
                            rotate(bound_plus, bound_minus, bound, alpha, phi);
                        }
                    }
                }
            }
            if ((action & POST_SHIFT) != 0) {
                shift(fplus, fminus);
                if constexpr (PAIRED) {
                    shift(bound_plus, bound_minus);
                }
            }
            if ((action & SPOIL_AFTER) != 0) {
                std::fill(fplus.begin(), fplus.end(), Complex{});
                std::fill(fminus.begin(), fminus.end(), Complex{});
                if constexpr (PAIRED) {
                    std::fill(bound_plus.begin(), bound_plus.end(), Complex{});
                    std::fill(bound_minus.begin(), bound_minus.end(), Complex{});
                }
            } else if ((action & SHIFT_AFTER) != 0) {
                shift(fplus, fminus);
                if constexpr (PAIRED) {
                    shift(bound_plus, bound_minus);
                }
            }
        }

        // ---- reverse ----
        std::fill(fplus_bar.begin(), fplus_bar.end(), Complex{});
        std::fill(fminus_bar.begin(), fminus_bar.end(), Complex{});
        std::fill(longitudinal_bar.begin(), longitudinal_bar.end(), Complex{});
        if constexpr (TWO_POOL || THREE) {
            std::fill(bound_bar.begin(), bound_bar.end(), Complex{});
        }
        if constexpr (THREE) {
            std::fill(semisolid_bar.begin(), semisolid_bar.end(), Complex{});
        }
        float grad_semisolid_fraction = 0.0F;
        float grad_semisolid_exchange = 0.0F;
        float grad_t1_semisolid = 0.0F;
        if constexpr (PAIRED) {
            std::fill(bound_plus_bar.begin(), bound_plus_bar.end(), Complex{});
            std::fill(bound_minus_bar.begin(), bound_minus_bar.end(), Complex{});
        }
        float grad_t2_bound = 0.0F;
        float grad_pool_shift = 0.0F;
        float grad_t1_bound = 0.0F;
        float grad_exchange = 0.0F;
        float grad_bound_fraction = 0.0F;
        float grad_t1 = 0.0F;
        float grad_t2 = 0.0F;
        float grad_m0 = 0.0F;
        float grad_b1 = 0.0F;
        float grad_b1_phase = 0.0F;
        // Transmit gradients are summed per shim: the running pair is flushed
        // to its row whenever the walk back reaches a pulse on a different
        // one. A single-shim sequence never changes row and flushes once.
        std::int64_t held = 0;
        float grad_b0 = 0.0F;
        float grad_efficiency = 0.0F;
        float grad_damping = 0.0F;
        float grad_flow = 0.0F;
        float grad_washout = 0.0F;

        for (std::int64_t event = event_count - 1; event >= 0; --event) {
            const Complex* slot = trajectory.data()
                + static_cast<std::size_t>(event) * trajectory_stride;
            std::copy(slot, slot + states, fplus.begin());
            std::copy(slot + states, slot + 2U * states, fminus.begin());
            std::copy(slot + 2U * states, slot + 3U * states, longitudinal.begin());
            if constexpr (TWO_POOL || THREE) {
                std::copy(slot + 3U * states, slot + 4U * states, bound.begin());
            }
            if constexpr (PAIRED) {
                std::copy(
                    slot + 4U * states, slot + 5U * states, bound_plus.begin()
                );
                std::copy(
                    slot + 5U * states, slot + 6U * states, bound_minus.begin()
                );
            }
            if constexpr (THREE) {
                std::copy(
                    slot + 6U * states, slot + 7U * states, semisolid.begin()
                );
            }

            const float dt = view.duration[event];
            damping.set(damping_rate, dt);
            const float dry1 = std::exp(-r1 * dt);
            const float dry2 = std::exp(-r2 * dt);
            const float wout = washout_out(washout_rate, dt);
            const float e1 = dry1 * wout;
            const float e2 = dry2 * wout;
            const TwoPoolStep<float> pools = TWO_POOL
                ? two_pool_step(r1, r1_bound, exchange, bound_fraction, dt, wout)
                : TwoPoolStep<float>{};
            const ThreePoolStep<double> triple = THREE
                ? three_pool_step<double>(
                    r1, r1_bound, r1_semisolid, exchange, semisolid_exchange,
                    bound_fraction, semisolid_fraction, dt, wout
                )
                : ThreePoolStep<double>{};
            const TwoPoolTransverse<Complex> across = PAIRED
                ? two_pool_transverse_step<float, Complex>(
                    r2, r2_bound, exchange, bound_fraction, transverse_free, pool_shift, dt, wout
                )
                : TwoPoolTransverse<Complex>{};
            const float angle = -2.0F * PI * b0 * dt;
            const Complex phase = turned(1.0F, angle, primal.off_axis);
            const std::uint8_t action = primal.action[event];

            // replay this event to recover the intra-event states
            for (std::size_t state = 0; state < states; ++state) {
                float turn_longitudinal = 0.0F;
                float turn_transverse = 0.0F;
                if (flowing) {
                    flow_turn(
                        flow_rate, dt, state, turn_longitudinal, turn_transverse
                    );
                }
                if constexpr (PAIRED) {
                    const Complex carried = turned(
                        damping.transverse[state], angle + turn_transverse, turning
                    );
                    const Complex free_plus = fplus[state];
                    const Complex pool_plus = bound_plus[state];
                    const Complex free_minus = fminus[state];
                    const Complex pool_minus = bound_minus[state];
                    fplus_relaxed[state] =
                        (across.e11 * free_plus + across.e12 * pool_plus) * carried;
                    bound_plus_relaxed[state] =
                        (across.e21 * free_plus + across.e22 * pool_plus) * carried;
                    const Complex conjugated = std::conj(carried);
                    fminus_relaxed[state] = (std::conj(across.e11) * free_minus
                        + std::conj(across.e12) * pool_minus) * conjugated;
                    bound_minus_relaxed[state] = (std::conj(across.e21) * free_minus
                        + std::conj(across.e22) * pool_minus) * conjugated;
                } else {
                    const Complex transverse = turned(
                        e2 * damping.transverse[state], angle + turn_transverse,
                        turning
                    );
                    fplus_relaxed[state] = fplus[state] * transverse;
                    fminus_relaxed[state] = fminus[state] * std::conj(transverse);
                }
                const Complex spin = turned(
                    damping.longitudinal[state], turn_longitudinal, flowing
                );
                if constexpr (THREE) {
                    const Complex pools_in[3] = {
                        longitudinal[state], bound[state], semisolid[state]
                    };
                    Complex mixed[3];
                    for (int row = 0; row < 3; ++row) {
                        Complex carried{};
                        for (int column = 0; column < 3; ++column) {
                            carried += static_cast<float>(triple.entry[row][column])
                                * pools_in[column];
                        }
                        mixed[row] = carried * spin;
                    }
                    longitudinal_relaxed[state] = mixed[0];
                    bound_relaxed[state] = mixed[1];
                    semisolid_relaxed[state] = mixed[2];
                } else if constexpr (TWO_POOL) {
                    const Complex free_state = longitudinal[state];
                    const Complex bound_state = bound[state];
                    longitudinal_relaxed[state] =
                        (pools.e11 * free_state + pools.e12 * bound_state) * spin;
                    bound_relaxed[state] =
                        (pools.e21 * free_state + pools.e22 * bound_state) * spin;
                } else {
                    longitudinal_relaxed[state] = longitudinal[state] * e1 * spin;
                }
            }
            if constexpr (THREE) {
                longitudinal_relaxed[0] +=
                    Complex(static_cast<float>(triple.recovery[0]), 0.0F);
                bound_relaxed[0] +=
                    Complex(static_cast<float>(triple.recovery[1]), 0.0F);
                semisolid_relaxed[0] +=
                    Complex(static_cast<float>(triple.recovery[2]), 0.0F);
            } else if constexpr (TWO_POOL) {
                longitudinal_relaxed[0] += Complex(pools.recovery_free, 0.0F);
                bound_relaxed[0] += Complex(pools.recovery_bound, 0.0F);
            } else {
                longitudinal_relaxed[0] += Complex(1.0F - e1, 0.0F);
            }

            fplus_shifted = fplus_relaxed;
            fminus_shifted = fminus_relaxed;
            longitudinal_pre = longitudinal_relaxed;
            if constexpr (PAIRED) {
                bound_plus_shifted = bound_plus_relaxed;
                bound_minus_shifted = bound_minus_relaxed;
            }
            if ((action & PRE_SHIFT) != 0) {
                shift(fplus_shifted, fminus_shifted);
                if constexpr (PAIRED) {
                    shift(bound_plus_shifted, bound_minus_shifted);
                }
            }

            // --- adjoint of the trailing shift/spoil ---
            if ((action & SPOIL_AFTER) != 0) {
                std::fill(fplus_bar.begin(), fplus_bar.end(), Complex{});
                std::fill(fminus_bar.begin(), fminus_bar.end(), Complex{});
                if constexpr (PAIRED) {
                    std::fill(
                        bound_plus_bar.begin(), bound_plus_bar.end(), Complex{}
                    );
                    std::fill(
                        bound_minus_bar.begin(), bound_minus_bar.end(), Complex{}
                    );
                }
            } else if ((action & SHIFT_AFTER) != 0) {
                shift_adjoint(fplus_bar, fminus_bar);
                if constexpr (PAIRED) {
                    shift_adjoint(bound_plus_bar, bound_minus_bar);
                }
            }
            if ((action & POST_SHIFT) != 0) {
                shift_adjoint(fplus_bar, fminus_bar);
                if constexpr (PAIRED) {
                    shift_adjoint(bound_plus_bar, bound_minus_bar);
                }
            }

            // --- adjoint of the ADC ---
            if (primal.kind[event] == 2 && (action & RECORD) != 0) {
                const std::int64_t output = primal.output_index[event];
                const std::int64_t index = view.output_base + output;
                const Complex seed(
                    buffers.grad_output_real[index],
                    buffers.grad_output_imag[index]
                );
                const Complex demodulation = std::polar(1.0F, -view.phase[event]);
                // an ADC event carries no RF, so the recorded state is the one
                // left by the pre-shift
                const Complex recorded = PAIRED
                    ? fplus_shifted[0] + bound_plus_shifted[0]
                    : fplus_shifted[0];
                grad_m0 += std::real(std::conj(seed) * recorded * demodulation);
                grad_phase_train[event] += std::real(
                    std::conj(seed) * m0 * recorded * Complex(0.0F, -1.0F)
                        * demodulation
                );
                const Complex weighted = std::conj(m0 * demodulation) * seed;
                fplus_bar[0] += weighted;
                if constexpr (PAIRED) {
                    bound_plus_bar[0] += weighted;
                }
            }

            // --- adjoint of the RF ---
            if (primal.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    for (std::size_t state = 0; state < states; ++state) {
                        grad_efficiency += std::real(
                            std::conj(longitudinal_bar[state])
                            * (-longitudinal_pre[state])
                        );
                        longitudinal_bar[state] *= -efficiency;
                        if constexpr (PAIRED) {
                            grad_efficiency += std::real(
                                std::conj(bound_bar[state]) * (-bound_relaxed[state])
                            );
                            bound_bar[state] *= -efficiency;
                        }
                    }
                } else {
                    const std::int64_t transmit =
                        shimmed ? transmit_row(primal, event, atom) : atom;
                    const float pulse_b1 =
                        shimmed ? primal.b1[transmit] : b1;
                    const float alpha = view.flip[event] * pulse_b1;
                    const float phi = view.phase[event]
                        + (shimmed ? primal.b1_phase[transmit] : b1_phase);
                    const std::int64_t row =
                        shimmed ? primal.shim_index[event] : 0;
                    if (row != held) {
                        grad_tissue_local[
                            (layout.base[B1_INDEX] + held) * atoms + atom
                        ] += grad_b1;
                        grad_tissue_local[
                            (layout.base[B1_PHASE_INDEX] + held) * atoms + atom
                        ] += grad_b1_phase;
                        grad_b1 = 0.0F;
                        grad_b1_phase = 0.0F;
                        held = row;
                    }
                    float grad_alpha = 0.0F;
                    float grad_phi = 0.0F;
                    if constexpr (MODE != RfMode::INSTANT) {
                        Complex pair_a{};
                        Complex pair_b{};
                        Complex slope_a{};
                        Complex slope_b{};
                        if constexpr (MODE == RfMode::DYNAMIC) {
                            dynamic_pair_at(
                                primal,
                                dynamic_row(primal, view, event),
                                atom, pair_a, pair_b
                            );
                        } else {
                            profile_pair_slope(
                                primal, table_row<MODE>(primal, event, location),
                                alpha, pair_a, pair_b, slope_a, slope_b
                            );
                        }
                        const Complex turn = std::polar(1.0F, -phi);
                        const Complex spun = pair_b * turn;
                        Complex grad_a{};
                        Complex grad_b{};
                        rotate_adjoint_spinor(
                            fplus_shifted,
                            fminus_shifted,
                            longitudinal_pre,
                            fplus_bar,
                            fminus_bar,
                            longitudinal_bar,
                            pair_a,
                            spun,
                            grad_a,
                            grad_b
                        );
                        if constexpr (PAIRED) {
                            rotate_adjoint_spinor(
                                bound_plus_shifted,
                                bound_minus_shifted,
                                bound_relaxed,
                                bound_plus_bar,
                                bound_minus_bar,
                                bound_bar,
                                pair_a,
                                spun,
                                grad_a,
                                grad_b
                            );
                        }
                        // The RF phase turns the axis once the pair is out,
                        // so it reaches ``b`` alone -- under either mode.
                        grad_phi = std::real(
                            std::conj(grad_b) * Complex(0.0F, -1.0F) * spun
                        );
                        if constexpr (MODE == RfMode::DYNAMIC) {
                            // The flip is inside the pair rather than read
                            // against it, so it has no gradient here: the
                            // cotangent goes out on the pair, and whatever
                            // integrated it carries the rest.
                            float* const entry = buffers.grad_dynamic
                                + dynamic_offset(
                                    primal,
                                    dynamic_row(primal, view, event),
                                    atom
                                );
                            // ``b`` was turned by the phase after the pair came
                            // out, so the cotangent turns back the other way.
                            const Complex held = grad_b * std::conj(turn);
                            entry[0] += grad_a.real();
                            entry[1] += grad_a.imag();
                            entry[2] += held.real();
                            entry[3] += held.imag();
                        } else {
                            // The flip angle reaches the pair through the slope
                            // the table stores beside it.
                            grad_alpha =
                                std::real(std::conj(grad_a) * slope_a)
                                + std::real(std::conj(grad_b) * slope_b * turn);
                        }
                    } else {
                        rotate_adjoint(
                            fplus_shifted,
                            fminus_shifted,
                            longitudinal_pre,
                            fplus_bar,
                            fminus_bar,
                            longitudinal_bar,
                            alpha,
                            phi,
                            grad_alpha,
                            grad_phi
                        );
                        if constexpr (PAIRED) {
                            rotate_adjoint(
                                bound_plus_shifted,
                                bound_minus_shifted,
                                bound_relaxed,
                                bound_plus_bar,
                                bound_minus_bar,
                                bound_bar,
                                alpha,
                                phi,
                                grad_alpha,
                                grad_phi
                            );
                        }
                    }
                    if constexpr (SATURATED) {
                        // The pulse scales every order of the semisolid pool by
                        // one real number, so its cotangent is a single sum over
                        // the states it multiplied.
                        State& absorbing = THREE ? semisolid : bound;
                        State& absorbing_bar = THREE ? semisolid_bar : bound_bar;
                        State& absorbing_relaxed =
                            THREE ? semisolid_relaxed : bound_relaxed;
                        const float offset = primal.rf_frequency[event] - b0;
                        float shape = 0.0F;
                        float shape_slope = 0.0F;
                        lineshape_at_slope(primal, offset, shape, shape_slope);
                        const float exponent =
                            primal.saturation[event] * alpha * alpha * shape;
                        const float absorbed = std::exp(exponent);
                        float grad_absorbed = 0.0F;
                        for (std::size_t state = 0; state < states; ++state) {
                            grad_absorbed += std::real(
                                std::conj(absorbing_bar[state])
                                * absorbing_relaxed[state]
                            );
                            absorbing_bar[state] *= absorbed;
                        }
                        (void)absorbing;
                        const float grad_exponent = grad_absorbed * absorbed;
                        grad_alpha += grad_exponent * primal.saturation[event]
                            * 2.0F * alpha * shape;
                        // The lineshape is read at the pulse's offset from the
                        // voxel, so a step in the voxel's own off-resonance
                        // moves the read the other way.
                        grad_b0 -= grad_exponent * primal.saturation[event]
                            * alpha * alpha * shape_slope;
                    }
                    grad_flip_train[event] += grad_alpha * pulse_b1;
                    grad_b1 += grad_alpha * view.flip[event];
                    grad_phase_train[event] += grad_phi;
                    grad_b1_phase += grad_phi;
                }
            }

            if ((action & PRE_SHIFT) != 0) {
                shift_adjoint(fplus_bar, fminus_bar);
                if constexpr (PAIRED) {
                    shift_adjoint(bound_plus_bar, bound_minus_bar);
                }
            }

            // --- adjoint of relaxation and precession ---
            float grad_e1 = 0.0F;
            float grad_e2 = 0.0F;
            float grad_angle = 0.0F;
            // The b-factor reaches every order through a different weight, so
            // its gradient is a separate weighted sum rather than a rescaling
            // of the relaxation gradients.
            float grad_b_factor = 0.0F;
            // Flow reaches each order through a turn of its own, so like the
            // b-factor it collects a weighted sum rather than a scalar one.
            float grad_turn = 0.0F;
            // The four entries of the exchange operator and the two recoveries,
            // summed over the orders that share them, then pushed back through
            // the closed form once for the whole interval.
            float grad_e11 = 0.0F;
            float grad_e12 = 0.0F;
            float grad_e21 = 0.0F;
            float grad_e22 = 0.0F;
            // The transverse operator's four entries, collected the same way.
            // They stay complex to the end: the map from the generator to them
            // is holomorphic, so nothing is projected onto the real line until
            // the sweep reaches the properties themselves.
            Complex grad_across_11{};
            Complex grad_across_12{};
            Complex grad_across_21{};
            Complex grad_across_22{};
            const Complex imaginary(0.0F, 1.0F);
            // Z[0] also carries the affine recovery term (1 - e1), whose
            // derivative contributes -Re(adjoint) exactly once. Order zero is
            // undamped, so that term takes the bare factor.
            const float grad_recovery_free =
                (TWO_POOL || THREE) ? std::real(longitudinal_bar[0]) : 0.0F;
            const float grad_recovery_bound =
                (TWO_POOL || THREE) ? std::real(bound_bar[0]) : 0.0F;
            const float grad_recovery_semisolid =
                THREE ? std::real(semisolid_bar[0]) : 0.0F;
            // The nine entries of the three-pool operator, summed over the
            // orders that share them and pushed back through the closed form
            // once for the whole interval.
            double grad_triple[3][3]{};
            if constexpr (!TWO_POOL && !THREE) {
                grad_e1 -= std::real(longitudinal_bar[0]);
            }
            for (std::size_t state = 0; state < states; ++state) {
                const Complex ap = fplus_bar[state];
                const Complex am = fminus_bar[state];
                const Complex az = longitudinal_bar[state];
                const float damp_transverse = damping.transverse[state];
                const float damp_longitudinal = damping.longitudinal[state];
                float turn_longitudinal = 0.0F;
                float turn_transverse = 0.0F;
                if (flowing) {
                    flow_turn(
                        flow_rate, dt, state, turn_longitudinal, turn_transverse
                    );
                }
                const Complex spin_transverse =
                    turned(1.0F, turn_transverse, flowing);
                const Complex spin_longitudinal =
                    turned(1.0F, turn_longitudinal, flowing);
                const Complex carried = phase * damp_transverse * spin_transverse;
                const Complex full_transverse = e2 * carried;
                // The damping is homogeneous of degree one in every transverse
                // state it acts on, so its gradient times the damping itself is
                // the cotangent taken against the states the interval leaves.
                float transverse_scaled = 0.0F;
                float angle_term = 0.0F;
                if constexpr (PAIRED) {
                    const Complex abp = bound_plus_bar[state];
                    const Complex abm = bound_minus_bar[state];
                    const Complex fp = fplus[state];
                    const Complex bp = bound_plus[state];
                    const Complex fm = fminus[state];
                    const Complex bm = bound_minus[state];
                    const Complex out_fp = fplus_relaxed[state];
                    const Complex out_bp = bound_plus_relaxed[state];
                    const Complex out_fm = fminus_relaxed[state];
                    const Complex out_bm = bound_minus_relaxed[state];
                    transverse_scaled = std::real(
                        std::conj(ap) * out_fp + std::conj(abp) * out_bp
                        + std::conj(am) * out_fm + std::conj(abm) * out_bm
                    );
                    angle_term = std::real(
                        imaginary * (std::conj(ap) * out_fp
                            + std::conj(abp) * out_bp
                            - std::conj(am) * out_fm
                            - std::conj(abm) * out_bm)
                    );
                    // ``F-`` follows the conjugate of the operator, so its
                    // cotangent lands on the entry itself rather than on the
                    // conjugate of it.
                    grad_across_11 +=
                        (std::conj(ap) * fp + am * std::conj(fm)) * carried;
                    grad_across_12 +=
                        (std::conj(ap) * bp + am * std::conj(bm)) * carried;
                    grad_across_21 +=
                        (std::conj(abp) * fp + abm * std::conj(fm)) * carried;
                    grad_across_22 +=
                        (std::conj(abp) * bp + abm * std::conj(bm)) * carried;
                    const Complex step_11 = across.e11 * carried;
                    const Complex step_12 = across.e12 * carried;
                    const Complex step_21 = across.e21 * carried;
                    const Complex step_22 = across.e22 * carried;
                    fplus_bar[state] =
                        std::conj(step_11) * ap + std::conj(step_21) * abp;
                    bound_plus_bar[state] =
                        std::conj(step_12) * ap + std::conj(step_22) * abp;
                    fminus_bar[state] = step_11 * am + step_21 * abm;
                    bound_minus_bar[state] = step_12 * am + step_22 * abm;
                } else {
                    const float transverse_term = std::real(
                        std::conj(ap) * carried * fplus[state]
                        + std::conj(am) * std::conj(carried) * fminus[state]
                    );
                    grad_e2 += transverse_term;
                    transverse_scaled = e2 * transverse_term;
                    angle_term = std::real(
                        std::conj(ap) * imaginary * full_transverse * fplus[state]
                        - std::conj(am) * imaginary * std::conj(full_transverse)
                            * fminus[state]
                    );
                    fplus_bar[state] = std::conj(full_transverse) * ap;
                    fminus_bar[state] = full_transverse * am;
                }
                // A turn of the transverse states and the off-resonance angle
                // are the same derivative; only the weight they carry differs.
                grad_angle += angle_term;
                // The damping carries the b-factor and the turn carries the
                // flow; both terms are taken against the state the interval
                // leaves, so a second pool changes what is summed, not how.
                const Complex spin = damp_longitudinal * spin_longitudinal;
                float longitudinal_damp_term = 0.0F;
                float longitudinal_angle_term = 0.0F;
                if constexpr (THREE) {
                    const Complex bars[3] = {az, bound_bar[state], semisolid_bar[state]};
                    const Complex pools_in[3] = {
                        longitudinal[state], bound[state], semisolid[state]
                    };
                    Complex mixed[3];
                    for (int row = 0; row < 3; ++row) {
                        Complex carried{};
                        for (int column = 0; column < 3; ++column) {
                            carried += static_cast<float>(triple.entry[row][column])
                                * pools_in[column];
                            grad_triple[row][column] += static_cast<double>(
                                std::real(std::conj(bars[row]) * spin * pools_in[column])
                            );
                        }
                        mixed[row] = carried;
                    }
                    for (int row = 0; row < 3; ++row) {
                        longitudinal_damp_term += std::real(
                            std::conj(bars[row]) * spin * mixed[row]
                        );
                        longitudinal_angle_term += std::real(
                            std::conj(bars[row]) * imaginary * spin * mixed[row]
                        );
                    }
                    Complex back[3];
                    for (int column = 0; column < 3; ++column) {
                        Complex carried{};
                        for (int row = 0; row < 3; ++row) {
                            carried += std::conj(
                                static_cast<float>(triple.entry[row][column]) * spin
                            ) * bars[row];
                        }
                        back[column] = carried;
                    }
                    longitudinal_bar[state] = back[0];
                    bound_bar[state] = back[1];
                    semisolid_bar[state] = back[2];
                } else if constexpr (TWO_POOL) {
                    const Complex ab = bound_bar[state];
                    const Complex free_state = longitudinal[state];
                    const Complex bound_state = bound[state];
                    const Complex mixed_free =
                        pools.e11 * free_state + pools.e12 * bound_state;
                    const Complex mixed_bound =
                        pools.e21 * free_state + pools.e22 * bound_state;
                    grad_e11 += std::real(std::conj(az) * spin * free_state);
                    grad_e12 += std::real(std::conj(az) * spin * bound_state);
                    grad_e21 += std::real(std::conj(ab) * spin * free_state);
                    grad_e22 += std::real(std::conj(ab) * spin * bound_state);
                    longitudinal_damp_term = std::real(
                        std::conj(az) * spin * mixed_free
                        + std::conj(ab) * spin * mixed_bound
                    );
                    longitudinal_angle_term = std::real(
                        std::conj(az) * imaginary * spin * mixed_free
                        + std::conj(ab) * imaginary * spin * mixed_bound
                    );
                    longitudinal_bar[state] = std::conj(pools.e11 * spin) * az
                        + std::conj(pools.e21 * spin) * ab;
                    bound_bar[state] = std::conj(pools.e12 * spin) * az
                        + std::conj(pools.e22 * spin) * ab;
                } else {
                    const Complex full_longitudinal = e1 * spin;
                    const float longitudinal_term =
                        std::real(std::conj(az) * spin * longitudinal[state]);
                    grad_e1 += longitudinal_term;
                    longitudinal_damp_term = e1 * longitudinal_term;
                    longitudinal_angle_term = std::real(
                        std::conj(az) * imaginary * full_longitudinal
                            * longitudinal[state]
                    );
                    longitudinal_bar[state] = std::conj(full_longitudinal) * az;
                }
                grad_b_factor -= transverse_scaled * transverse_weight(state)
                    + longitudinal_damp_term * longitudinal_weight(state);
                grad_turn -= angle_term * (static_cast<float>(state) + 0.5F)
                    + longitudinal_angle_term * static_cast<float>(state);
            }

            float grad_exchange_attenuation = 0.0F;
            float grad_two_pool_duration = 0.0F;
            if constexpr (TWO_POOL) {
                const TwoPoolGradient<float> back = two_pool_step_adjoint(
                    r1, r1_bound, exchange, bound_fraction, dt, wout,
                    grad_e11, grad_e12, grad_e21, grad_e22,
                    grad_recovery_free, grad_recovery_bound
                );
                grad_t1 += back.r1_free * (-1000.0F / (t1 * t1));
                grad_t1_bound += back.r1_bound * (-1000.0F / (t1_bound * t1_bound));
                grad_exchange += back.exchange;
                grad_bound_fraction += back.bound;
                grad_exchange_attenuation = back.attenuation;
                grad_two_pool_duration = back.dt;
            }
            if constexpr (THREE) {
                double bar_entry[3][3];
                for (int row = 0; row < 3; ++row) {
                    for (int column = 0; column < 3; ++column) {
                        bar_entry[row][column] = grad_triple[row][column];
                    }
                }
                const double bar_recovery[3] = {
                    static_cast<double>(grad_recovery_free),
                    static_cast<double>(grad_recovery_bound),
                    static_cast<double>(grad_recovery_semisolid),
                };
                const ThreePoolGradient<double> back = three_pool_step_adjoint<double>(
                    r1, r1_bound, r1_semisolid, exchange, semisolid_exchange,
                    bound_fraction, semisolid_fraction, dt, wout,
                    bar_entry, bar_recovery
                );
                grad_t1 += static_cast<float>(back.r1_free) * (-1000.0F / (t1 * t1));
                grad_t1_bound += static_cast<float>(back.r1_pool_b)
                    * (-1000.0F / (t1_bound * t1_bound));
                grad_t1_semisolid += static_cast<float>(back.r1_bound)
                    * (-1000.0F / (t1_semisolid * t1_semisolid));
                grad_exchange += static_cast<float>(back.exchange_b);
                grad_semisolid_exchange += static_cast<float>(back.exchange_c);
                grad_bound_fraction += static_cast<float>(back.fraction_b);
                grad_semisolid_fraction += static_cast<float>(back.fraction_c);
                grad_exchange_attenuation += static_cast<float>(back.attenuation);
                grad_two_pool_duration += static_cast<float>(back.dt);
            }
            if constexpr (PAIRED) {
                const TwoPoolTransverseGradient<float> across_back =
                    two_pool_transverse_adjoint<float, Complex>(
                        r2, r2_bound, exchange, bound_fraction, transverse_free,
                        pool_shift, dt, wout, grad_across_11, grad_across_12,
                        grad_across_21, grad_across_22
                    );
                grad_t2 += across_back.r2_free * (-1000.0F / (t2 * t2));
                grad_t2_bound +=
                    across_back.r2_bound * (-1000.0F / (t2_bound * t2_bound));
                grad_exchange += across_back.exchange;
                // The free water is what both second pools leave, so a
                // cotangent on it reaches each of their fractions turned over.
                grad_bound_fraction += across_back.bound - across_back.free;
                grad_semisolid_fraction -= THREE ? across_back.free : 0.0F;
                grad_pool_shift += across_back.shift_hz;
                grad_exchange_attenuation += across_back.attenuation;
                grad_two_pool_duration += across_back.dt;
            }

            // Washout scales both relaxation factors, so its gradient is the
            // one they already carry, taken against those factors as they
            // stand before that scaling. Past the clamp nothing depends on the
            // rate any more.
            const float grad_wout = washout_rate * dt < 1.0F
                ? -(dry1 * grad_e1 + dry2 * grad_e2 + grad_exchange_attenuation)
                : 0.0F;

            grad_t1 += grad_e1 * e1 * 1000.0F * dt / (t1 * t1);
            grad_t2 += grad_e2 * e2 * 1000.0F * dt / (t2 * t2);
            grad_b0 += grad_angle * (-2.0F * PI * dt);
            grad_damping += grad_b_factor * dt;
            grad_flow += grad_turn * dt;
            grad_washout += grad_wout * dt;
            grad_duration_train[event] +=
                grad_e1 * (-r1 * e1) + grad_e2 * (-r2 * e2)
                + grad_two_pool_duration
                + grad_angle * (-2.0F * PI * b0)
                + grad_b_factor * damping_rate
                + grad_turn * flow_rate
                + grad_wout * washout_rate;
        }

        if constexpr (TWO_POOL || THREE) {
            // The fraction also sets where each pool starts, which the walk
            // back reaches last.
            grad_bound_fraction +=
                std::real(bound_bar[0]) - std::real(longitudinal_bar[0]);
        }
        if constexpr (THREE) {
            grad_semisolid_fraction +=
                std::real(semisolid_bar[0]) - std::real(longitudinal_bar[0]);
        }
        if constexpr (MT) {
            grad_tissue_local[layout.base[9] * atoms + atom] += grad_bound_fraction;
            grad_tissue_local[layout.base[10] * atoms + atom] += grad_exchange;
            grad_tissue_local[layout.base[11] * atoms + atom] += grad_t1_bound;
        }
        if constexpr (THREE) {
            grad_tissue_local[layout.base[9] * atoms + atom] +=
                grad_semisolid_fraction;
            grad_tissue_local[layout.base[10] * atoms + atom] +=
                grad_semisolid_exchange;
            grad_tissue_local[layout.base[11] * atoms + atom] += grad_t1_semisolid;
        }
        if constexpr (PAIRED) {
            grad_tissue_local[layout.base[12] * atoms + atom] += grad_bound_fraction;
            grad_tissue_local[layout.base[13] * atoms + atom] += grad_exchange;
            grad_tissue_local[layout.base[14] * atoms + atom] += grad_t1_bound;
            grad_tissue_local[layout.base[15] * atoms + atom] += grad_t2_bound;
            grad_tissue_local[layout.base[16] * atoms + atom] += grad_pool_shift;
        }
        grad_tissue_local[layout.base[0] * atoms + atom] += grad_t1;
        grad_tissue_local[layout.base[1] * atoms + atom] += grad_t2;
        grad_tissue_local[layout.base[2] * atoms + atom] += grad_m0;
        grad_tissue_local[(layout.base[B1_INDEX] + held) * atoms + atom] += grad_b1;
        grad_tissue_local[(layout.base[B1_PHASE_INDEX] + held) * atoms + atom] +=
            grad_b1_phase;
        grad_tissue_local[layout.base[5] * atoms + atom] += grad_b0;
        grad_tissue_local[layout.base[6] * atoms + atom] += grad_efficiency;
        grad_tissue_local[layout.base[7] * atoms + atom] += grad_damping;
        // One buffer drives two rates, so the velocity gradient is the sum of
        // what each geometry carries back.
        grad_tissue_local[layout.base[8] * atoms + atom] +=
            grad_flow * primal.flow_scale
            + grad_washout * speed_direction(velocity) * primal.washout_scale;
    }
}

inline void shift_real_adjoint(
    std::vector<float>& plus_bar, std::vector<float>& minus_bar,
    const std::size_t states
) {
    // No states is impossible -- every entry point rejects a state count
    // below one -- but not provable here, and the loops below count down
    // from ``states - 1``.
    if (states == 0) {
        return;
    }
    // ``shift_real`` ends with plus[0] = -minus[0], reading the minus vector
    // after its own shift -- so the entry it couples to is minus[1] as it
    // stood before, and the cotangent lands there.
    const float carry = -plus_bar[0];
    for (std::size_t state = 0; state + 1 < states; ++state) {
        plus_bar[state] = plus_bar[state + 1];
    }
    plus_bar[states - 1] = 0.0F;
    for (std::size_t state = states - 1; state > 0; --state) {
        minus_bar[state] = minus_bar[state - 1];
    }
    minus_bar[0] = 0.0F;
    if (states > 1) {
        minus_bar[1] += carry;
    }
}

// The adjoint through the real subspace.
//
// The forward is a chain of real linear maps, so each adjoint is a transpose
// rather than a conjugate transpose and the state carries three reals instead
// of three complex numbers -- the same saving the real forward makes, in the
// pass that costs the most.
//
// The RF phase, the transmit phase and off-resonance divide out of the
// representation, so this kernel leaves their gradients untouched; the flow
// winding does too. Callers must have established those conditions and must
// not ask for those four gradients; see ``real_subspace_axis`` and the
// ``wanted`` mask that guards it.
void simulate_real_vjp_range(
    const VjpBuffers& buffers,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count,
    float* grad_flip_local,
    float* grad_phase_local,
    float* grad_duration_local,
    float* grad_tissue_local
) {
    (void)grad_phase_local;
    const Buffers& primal = buffers.primal;
    const std::int64_t atoms = primal.atom_count;
    const std::size_t states = static_cast<std::size_t>(state_count);
    const std::size_t stride = 3U * states;

    std::vector<float> trajectory(
        static_cast<std::size_t>(event_count) * stride
    );
    std::vector<float> plus(states), minus(states), longitudinal(states);
    std::vector<float> plus_bar(states), minus_bar(states), longitudinal_bar(states);
    std::vector<float> plus_stage(states), minus_stage(states);
    std::vector<float> longitudinal_stage(states);
    Damping<float> damping(states);

    for (std::int64_t work = work_begin; work < work_end; ++work) {
        const TrainView view = train_view(primal, work, event_count, output_count);
        const std::int64_t atom = view.atom;
        float* const grad_flip_train = grad_flip_local + view.event_base;
        float* const grad_duration_train = grad_duration_local + view.event_base;

        const float t1 = primal.t1[atom];
        const float t2 = primal.t2[atom];
        const float m0 = primal.density ? primal.m0[atom] : 1.0F;
        const float b1 = primal.transmit ? primal.b1[atom] : 1.0F;
        const float inversion =
            primal.inverting ? primal.inversion_efficiency[atom] : 1.0F;
        const float r1 = 1000.0F / t1;
        const float r2 = 1000.0F / t2;
        const float damping_rate =
            primal.diffusing ? primal.diffusion[atom] : 0.0F;

        std::fill(plus.begin(), plus.end(), 0.0F);
        std::fill(minus.begin(), minus.end(), 0.0F);
        std::fill(longitudinal.begin(), longitudinal.end(), 0.0F);
        longitudinal[0] = 1.0F;

        // ---- forward, recording the state entering each event ----
        for (std::int64_t event = 0; event < event_count; ++event) {
            float* slot = trajectory.data()
                + static_cast<std::size_t>(event) * stride;
            std::copy(plus.begin(), plus.end(), slot);
            std::copy(minus.begin(), minus.end(), slot + states);
            std::copy(longitudinal.begin(), longitudinal.end(), slot + 2U * states);

            const float dt = view.duration[event];
            damping.set(damping_rate, dt);
            const float e1 = std::exp(-r1 * dt);
            const float e2 = std::exp(-r2 * dt);
            for (std::size_t state = 0; state < states; ++state) {
                const float damp_transverse = e2 * damping.transverse[state];
                plus[state] *= damp_transverse;
                minus[state] *= damp_transverse;
                longitudinal[state] *= e1 * damping.longitudinal[state];
            }
            longitudinal[0] += 1.0F - e1;

            const std::uint8_t action = primal.action[event];
            if ((action & PRE_SHIFT) != 0) {
                shift_real(plus, minus, states);
            }
            if (primal.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    const float efficiency = -inversion;
                    for (std::size_t state = 0; state < states; ++state) {
                        longitudinal[state] *= efficiency;
                    }
                } else {
                    const float alpha = view.flip[event] * b1;
                    const float cosine = std::cos(alpha);
                    const float sine = std::sin(alpha);
                    const float cosine_half_sq = 0.5F * (1.0F + cosine);
                    const float sine_half_sq = 0.5F * (1.0F - cosine);
                    const float half_sine = 0.5F * sine;
                    for (std::size_t state = 0; state < states; ++state) {
                        const float p = plus[state];
                        const float m = minus[state];
                        const float z = longitudinal[state];
                        plus[state] = cosine_half_sq * p + sine_half_sq * m - sine * z;
                        minus[state] = sine_half_sq * p + cosine_half_sq * m + sine * z;
                        longitudinal[state] = half_sine * p - half_sine * m + cosine * z;
                    }
                }
            }
            if ((action & POST_SHIFT) != 0) {
                shift_real(plus, minus, states);
            }
            if ((action & SPOIL_AFTER) != 0) {
                std::fill(plus.begin(), plus.end(), 0.0F);
                std::fill(minus.begin(), minus.end(), 0.0F);
            } else if ((action & SHIFT_AFTER) != 0) {
                shift_real(plus, minus, states);
            }
        }

        // ---- reverse ----
        std::fill(plus_bar.begin(), plus_bar.end(), 0.0F);
        std::fill(minus_bar.begin(), minus_bar.end(), 0.0F);
        std::fill(longitudinal_bar.begin(), longitudinal_bar.end(), 0.0F);
        float grad_t1 = 0.0F;
        float grad_t2 = 0.0F;
        float grad_m0 = 0.0F;
        float grad_b1 = 0.0F;
        float grad_inversion = 0.0F;
        float grad_damping = 0.0F;

        for (std::int64_t event = event_count - 1; event >= 0; --event) {
            const float* slot = trajectory.data()
                + static_cast<std::size_t>(event) * stride;
            const std::uint8_t action = primal.action[event];
            const float dt = view.duration[event];
            damping.set(damping_rate, dt);
            const float e1 = std::exp(-r1 * dt);
            const float e2 = std::exp(-r2 * dt);

            // Replay the intra-event stages from the recorded entry state.
            for (std::size_t state = 0; state < states; ++state) {
                const float damp_transverse = e2 * damping.transverse[state];
                plus_stage[state] = slot[state] * damp_transverse;
                minus_stage[state] = slot[states + state] * damp_transverse;
                longitudinal_stage[state] =
                    slot[2U * states + state] * e1 * damping.longitudinal[state];
            }
            longitudinal_stage[0] += 1.0F - e1;
            if ((action & PRE_SHIFT) != 0) {
                shift_real(plus_stage, minus_stage, states);
            }
            // plus_stage now holds the state entering the RF operator.

            // Undo the trailing spoil/shift.
            if ((action & SPOIL_AFTER) != 0) {
                std::fill(plus_bar.begin(), plus_bar.end(), 0.0F);
                std::fill(minus_bar.begin(), minus_bar.end(), 0.0F);
            } else if ((action & SHIFT_AFTER) != 0) {
                shift_real_adjoint(plus_bar, minus_bar, states);
            }
            if ((action & POST_SHIFT) != 0) {
                shift_real_adjoint(plus_bar, minus_bar, states);
            }

            if (primal.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    const float efficiency = -inversion;
                    for (std::size_t state = 0; state < states; ++state) {
                        grad_inversion -=
                            longitudinal_bar[state] * longitudinal_stage[state];
                        longitudinal_bar[state] *= efficiency;
                    }
                } else {
                    const float flip = view.flip[event];
                    const float alpha = flip * b1;
                    const float cosine = std::cos(alpha);
                    const float sine = std::sin(alpha);
                    const float cosine_half_sq = 0.5F * (1.0F + cosine);
                    const float sine_half_sq = 0.5F * (1.0F - cosine);
                    const float half_sine = 0.5F * sine;
                    float grad_alpha = 0.0F;
                    for (std::size_t state = 0; state < states; ++state) {
                        const float p = plus_stage[state];
                        const float m = minus_stage[state];
                        const float z = longitudinal_stage[state];
                        const float pb = plus_bar[state];
                        const float mb = minus_bar[state];
                        const float zb = longitudinal_bar[state];
                        // d/dalpha of each output row, contracted with the adjoint.
                        grad_alpha += pb * (half_sine * m - half_sine * p - cosine * z)
                            + mb * (half_sine * p - half_sine * m + cosine * z)
                            + zb * (0.5F * cosine * p - 0.5F * cosine * m - sine * z);
                        // Transpose of the rotation.
                        plus_bar[state] =
                            cosine_half_sq * pb + sine_half_sq * mb + half_sine * zb;
                        minus_bar[state] =
                            sine_half_sq * pb + cosine_half_sq * mb - half_sine * zb;
                        longitudinal_bar[state] =
                            -sine * pb + sine * mb + cosine * zb;
                    }
                    grad_flip_train[event] += grad_alpha * b1;
                    grad_b1 += grad_alpha * flip;
                }
            } else if (primal.kind[event] == 2 && (action & RECORD) != 0) {
                // The sample is i * m0 * plus[0]; only the imaginary seed acts.
                const std::int64_t index =
                    view.output_base + primal.output_index[event];
                const float seed = buffers.grad_output_imag[index];
                grad_m0 += seed * plus_stage[0];
                plus_bar[0] += seed * m0;
            }

            if ((action & PRE_SHIFT) != 0) {
                shift_real_adjoint(plus_bar, minus_bar, states);
            }

            float grad_e1 = -longitudinal_bar[0];
            float grad_e2 = 0.0F;
            float grad_b_factor = 0.0F;
            for (std::size_t state = 0; state < states; ++state) {
                const float damp_transverse = damping.transverse[state];
                const float damp_longitudinal = damping.longitudinal[state];
                const float plus_term =
                    plus_bar[state] * slot[state] * damp_transverse;
                const float minus_term =
                    minus_bar[state] * slot[states + state] * damp_transverse;
                const float longitudinal_term = longitudinal_bar[state]
                    * slot[2U * states + state] * damp_longitudinal;
                grad_e2 += plus_term + minus_term;
                grad_e1 += longitudinal_term;
                grad_b_factor -=
                    transverse_weight(state) * (e2 * (plus_term + minus_term))
                    + longitudinal_weight(state) * (e1 * longitudinal_term);
                plus_bar[state] *= e2 * damp_transverse;
                minus_bar[state] *= e2 * damp_transverse;
                longitudinal_bar[state] *= e1 * damp_longitudinal;
            }
            grad_t1 += grad_e1 * (e1 * dt * (1000.0F / (t1 * t1)));
            grad_t2 += grad_e2 * (e2 * dt * (1000.0F / (t2 * t2)));
            grad_damping += grad_b_factor * dt;
            grad_duration_train[event] += -grad_e1 * (r1 * e1) - grad_e2 * (r2 * e2)
                + grad_b_factor * damping_rate;
        }

        // A real-subspace run is a single-shim one, so each parameter's row is
        // its own index; the four the representation divides out are left
        // alone, which is what the caller promised not to read.
        grad_tissue_local[0 * atoms + atom] += grad_t1;
        grad_tissue_local[1 * atoms + atom] += grad_t2;
        grad_tissue_local[2 * atoms + atom] += grad_m0;
        grad_tissue_local[3 * atoms + atom] += grad_b1;
        grad_tissue_local[6 * atoms + atom] += grad_inversion;
        grad_tissue_local[7 * atoms + atom] += grad_damping;
    }
}


// The reverse of ``shift_real_lanes``, eight atoms wide.
inline void shift_real_adjoint_lanes(
    float* const plus_bar, float* const minus_bar, const std::size_t states
) {
    if (states == 0) {
        return;
    }
    const LaneVector carry = -lane_load(plus_bar);
    for (std::size_t state = 0; state + 1 < states; ++state) {
        lane_store(
            plus_bar + state * REAL_LANES,
            lane_load(plus_bar + (state + 1) * REAL_LANES)
        );
    }
    lane_store(plus_bar + (states - 1) * REAL_LANES, lane_splat(0.0F));
    for (std::size_t state = states - 1; state > 0; --state) {
        lane_store(
            minus_bar + state * REAL_LANES,
            lane_load(minus_bar + (state - 1) * REAL_LANES)
        );
    }
    lane_store(minus_bar, lane_splat(0.0F));
    if (states > 1) {
        lane_store(
            minus_bar + REAL_LANES, lane_load(minus_bar + REAL_LANES) + carry
        );
    }
}

// A gradient on an event value is one number every atom contributes to, so the
// block's lanes are summed across before it lands.
inline float lane_sum(const LaneVector value) {
    float buffer[REAL_LANES];
    lane_store(buffer, value);
    float total = 0.0F;
    for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
        total += buffer[lane];
    }
    return total;
}

// The adjoint of the real subspace, eight atoms at a time. The arithmetic is
// the per-atom kernel's, laid out the way ``simulate_real_jvp_atom_lane_range``
// lays the forward-mode pass out: the tissue is gathered once per block, every
// event value is a splat, and the trajectory is recorded lane-major.
//
// An inactive lane of a partial block is seeded with a zero cotangent, which
// leaves every gradient it computes at zero -- so the sums over lanes that
// carry an event's gradient need no mask, and the per-atom rows it would write
// are never scattered.
//
// The dispatch reaches this kernel only where no atom diffuses, so the damping
// factors are one throughout; the gradient along the damping rate is not, and
// is accumulated from the state weights as the per-atom kernel does.
void simulate_real_vjp_atom_lane_range(
    const VjpBuffers& buffers,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count,
    float* grad_flip_local,
    float* grad_phase_local,
    float* grad_duration_local,
    float* grad_tissue_local
) {
    (void)grad_phase_local;
    const Buffers& primal = buffers.primal;
    const std::int64_t atoms = primal.atom_count;
    const std::size_t states = static_cast<std::size_t>(state_count);
    const std::size_t width = states * REAL_LANES;
    const std::size_t stride = 3U * width;

    std::vector<float> trajectory(
        static_cast<std::size_t>(event_count) * stride
    );
    std::vector<float> storage(6U * width);
    float* const TORCHSIM_RESTRICT plus = storage.data();
    float* const TORCHSIM_RESTRICT minus = plus + width;
    float* const TORCHSIM_RESTRICT longitudinal = minus + width;
    float* const TORCHSIM_RESTRICT plus_bar = longitudinal + width;
    float* const TORCHSIM_RESTRICT minus_bar = plus_bar + width;
    float* const TORCHSIM_RESTRICT longitudinal_bar = minus_bar + width;

    float t1[REAL_LANES], t2[REAL_LANES], m0[REAL_LANES], b1[REAL_LANES];
    float efficiency[REAL_LANES];
    float rate1[REAL_LANES], rate2[REAL_LANES];
    float scale1[REAL_LANES], scale2[REAL_LANES];
    float e1[REAL_LANES], e2[REAL_LANES];
    float cosine[REAL_LANES], sine[REAL_LANES];
    float seed[REAL_LANES];
    const LaneVector half = lane_splat(0.5F);
    const LaneVector one = lane_splat(1.0F);
    const LaneVector zero = lane_splat(0.0F);

    for (std::int64_t work = work_begin; work < work_end; ++work) {
        const AtomLaneView view = atom_lane_view(primal, work);
        const std::int64_t event_base = view.train * event_count;
        float* const grad_flip_train = grad_flip_local + event_base;
        float* const grad_duration_train = grad_duration_local + event_base;

        gather_atoms(primal.t1, true, 1.0F, view, t1);
        gather_atoms(primal.t2, true, 1.0F, view, t2);
        gather_atoms(primal.m0, primal.density, 1.0F, view, m0);
        gather_atoms(primal.b1, primal.transmit, 1.0F, view, b1);
        gather_atoms(
            primal.inversion_efficiency, primal.inverting, 1.0F, view, efficiency
        );
        for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
            rate1[lane] = 1000.0F / t1[lane];
            rate2[lane] = 1000.0F / t2[lane];
            scale1[lane] = 1000.0F / (t1[lane] * t1[lane]);
            scale2[lane] = 1000.0F / (t2[lane] * t2[lane]);
        }
        const LaneVector v_m0 = lane_load(m0);
        const LaneVector v_b1 = lane_load(b1);
        const LaneVector v_inversion = lane_load(efficiency);
        const LaneVector v_rate1 = lane_load(rate1);
        const LaneVector v_rate2 = lane_load(rate2);
        const LaneVector t1_scale = lane_load(scale1);
        const LaneVector t2_scale = lane_load(scale2);

        std::fill(storage.begin(), storage.end(), 0.0F);
        for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
            longitudinal[lane] = 1.0F;
        }

        // ---- forward, recording the state entering each event ----
        for (std::int64_t event = 0; event < event_count; ++event) {
            float* const slot = trajectory.data()
                + static_cast<std::size_t>(event) * stride;
            std::copy(plus, plus + width, slot);
            std::copy(minus, minus + width, slot + width);
            std::copy(longitudinal, longitudinal + width, slot + 2U * width);

            const float dt = primal.duration[event_base + event];
            for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
                e1[lane] = std::exp(-rate1[lane] * dt);
                e2[lane] = std::exp(-rate2[lane] * dt);
            }
            const LaneVector v_e1 = lane_load(e1);
            const LaneVector v_e2 = lane_load(e2);
            for (std::size_t state = 0; state < states; ++state) {
                const std::size_t base = state * REAL_LANES;
                lane_store(plus + base, lane_load(plus + base) * v_e2);
                lane_store(minus + base, lane_load(minus + base) * v_e2);
                lane_store(
                    longitudinal + base, lane_load(longitudinal + base) * v_e1
                );
            }
            lane_store(longitudinal, lane_load(longitudinal) + (one - v_e1));

            const std::uint8_t action = primal.action[event];
            if ((action & PRE_SHIFT) != 0) {
                shift_real_lanes(plus, minus, states);
            }
            if (primal.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    const LaneVector factor = -v_inversion;
                    for (std::size_t state = 0; state < states; ++state) {
                        const std::size_t base = state * REAL_LANES;
                        lane_store(
                            longitudinal + base, lane_load(longitudinal + base) * factor
                        );
                    }
                } else {
                    const float flip = primal.flip[event_base + event];
                    for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
                        const float alpha = flip * b1[lane];
                        cosine[lane] = std::cos(alpha);
                        sine[lane] = std::sin(alpha);
                    }
                    const LaneVector c = lane_load(cosine);
                    const LaneVector s = lane_load(sine);
                    const LaneVector chs = half * (one + c);
                    const LaneVector shs = half * (one - c);
                    const LaneVector hs = half * s;
                    for (std::size_t state = 0; state < states; ++state) {
                        const std::size_t base = state * REAL_LANES;
                        const LaneVector p = lane_load(plus + base);
                        const LaneVector m = lane_load(minus + base);
                        const LaneVector z = lane_load(longitudinal + base);
                        lane_store(plus + base, chs * p + shs * m - s * z);
                        lane_store(minus + base, shs * p + chs * m + s * z);
                        lane_store(longitudinal + base, hs * p - hs * m + c * z);
                    }
                }
            }
            if ((action & POST_SHIFT) != 0) {
                shift_real_lanes(plus, minus, states);
            }
            if ((action & SPOIL_AFTER) != 0) {
                std::fill(plus, plus + width, 0.0F);
                std::fill(minus, minus + width, 0.0F);
            } else if ((action & SHIFT_AFTER) != 0) {
                shift_real_lanes(plus, minus, states);
            }
        }

        // ---- reverse ----
        std::fill(plus_bar, plus_bar + 3U * width, 0.0F);
        LaneVector grad_t1 = zero;
        LaneVector grad_t2 = zero;
        LaneVector grad_m0 = zero;
        LaneVector grad_b1 = zero;
        LaneVector grad_inversion = zero;
        LaneVector grad_damping = zero;

        for (std::int64_t event = event_count - 1; event >= 0; --event) {
            const float* const slot = trajectory.data()
                + static_cast<std::size_t>(event) * stride;
            const std::uint8_t action = primal.action[event];
            const float dt = primal.duration[event_base + event];
            for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
                e1[lane] = std::exp(-rate1[lane] * dt);
                e2[lane] = std::exp(-rate2[lane] * dt);
            }
            const LaneVector v_e1 = lane_load(e1);
            const LaneVector v_e2 = lane_load(e2);

            // Replay the intra-event stages from the recorded entry state, into
            // the state planes the forward pass has finished with.
            for (std::size_t state = 0; state < states; ++state) {
                const std::size_t base = state * REAL_LANES;
                lane_store(plus + base, lane_load(slot + base) * v_e2);
                lane_store(minus + base, lane_load(slot + width + base) * v_e2);
                lane_store(
                    longitudinal + base,
                    lane_load(slot + 2U * width + base) * v_e1
                );
            }
            lane_store(longitudinal, lane_load(longitudinal) + (one - v_e1));
            if ((action & PRE_SHIFT) != 0) {
                shift_real_lanes(plus, minus, states);
            }
            // The state planes now hold what entered the RF operator.

            if ((action & SPOIL_AFTER) != 0) {
                std::fill(plus_bar, plus_bar + width, 0.0F);
                std::fill(minus_bar, minus_bar + width, 0.0F);
            } else if ((action & SHIFT_AFTER) != 0) {
                shift_real_adjoint_lanes(plus_bar, minus_bar, states);
            }
            if ((action & POST_SHIFT) != 0) {
                shift_real_adjoint_lanes(plus_bar, minus_bar, states);
            }

            if (primal.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    const LaneVector factor = -v_inversion;
                    for (std::size_t state = 0; state < states; ++state) {
                        const std::size_t base = state * REAL_LANES;
                        const LaneVector zb = lane_load(longitudinal_bar + base);
                        grad_inversion =
                            grad_inversion - zb * lane_load(longitudinal + base);
                        lane_store(longitudinal_bar + base, zb * factor);
                    }
                } else {
                    const float flip = primal.flip[event_base + event];
                    for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
                        const float alpha = flip * b1[lane];
                        cosine[lane] = std::cos(alpha);
                        sine[lane] = std::sin(alpha);
                    }
                    const LaneVector c = lane_load(cosine);
                    const LaneVector s = lane_load(sine);
                    const LaneVector chs = half * (one + c);
                    const LaneVector shs = half * (one - c);
                    const LaneVector hs = half * s;
                    LaneVector grad_alpha = zero;
                    for (std::size_t state = 0; state < states; ++state) {
                        const std::size_t base = state * REAL_LANES;
                        const LaneVector p = lane_load(plus + base);
                        const LaneVector m = lane_load(minus + base);
                        const LaneVector z = lane_load(longitudinal + base);
                        const LaneVector pb = lane_load(plus_bar + base);
                        const LaneVector mb = lane_load(minus_bar + base);
                        const LaneVector zb = lane_load(longitudinal_bar + base);
                        grad_alpha = grad_alpha
                            + pb * (hs * m - hs * p - c * z)
                            + mb * (hs * p - hs * m + c * z)
                            + zb * (half * c * p - half * c * m - s * z);
                        lane_store(plus_bar + base, chs * pb + shs * mb + hs * zb);
                        lane_store(minus_bar + base, shs * pb + chs * mb - hs * zb);
                        lane_store(
                            longitudinal_bar + base, -(s * pb) + s * mb + c * zb
                        );
                    }
                    grad_flip_train[event] += lane_sum(grad_alpha * v_b1);
                    grad_b1 = grad_b1 + grad_alpha * lane_splat(flip);
                }
            } else if (primal.kind[event] == 2 && (action & RECORD) != 0) {
                // The sample is i * m0 * plus[0]; only the imaginary seed acts.
                const std::int64_t offset = primal.output_index[event];
                for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
                    if (static_cast<std::int64_t>(lane) >= view.active) {
                        seed[lane] = 0.0F;
                        continue;
                    }
                    const std::int64_t index =
                        (view.train * atoms + view.atom_begin
                         + static_cast<std::int64_t>(lane))
                            * output_count
                        + offset;
                    seed[lane] = buffers.grad_output_imag[index];
                }
                const LaneVector v_seed = lane_load(seed);
                grad_m0 = grad_m0 + v_seed * lane_load(plus);
                lane_store(plus_bar, lane_load(plus_bar) + v_seed * v_m0);
            }

            if ((action & PRE_SHIFT) != 0) {
                shift_real_adjoint_lanes(plus_bar, minus_bar, states);
            }

            LaneVector grad_e1 = -lane_load(longitudinal_bar);
            LaneVector grad_e2 = zero;
            LaneVector grad_b_factor = zero;
            for (std::size_t state = 0; state < states; ++state) {
                const std::size_t base = state * REAL_LANES;
                const LaneVector pb = lane_load(plus_bar + base);
                const LaneVector mb = lane_load(minus_bar + base);
                const LaneVector zb = lane_load(longitudinal_bar + base);
                const LaneVector plus_term = pb * lane_load(slot + base);
                const LaneVector minus_term = mb * lane_load(slot + width + base);
                const LaneVector longitudinal_term =
                    zb * lane_load(slot + 2U * width + base);
                grad_e2 = grad_e2 + plus_term + minus_term;
                grad_e1 = grad_e1 + longitudinal_term;
                grad_b_factor = grad_b_factor
                    - (lane_splat(transverse_weight(state))
                           * (v_e2 * (plus_term + minus_term))
                       + lane_splat(longitudinal_weight(state))
                           * (v_e1 * longitudinal_term));
                lane_store(plus_bar + base, pb * v_e2);
                lane_store(minus_bar + base, mb * v_e2);
                lane_store(longitudinal_bar + base, zb * v_e1);
            }
            const LaneVector v_dt = lane_splat(dt);
            grad_t1 = grad_t1 + grad_e1 * (v_e1 * v_dt * t1_scale);
            grad_t2 = grad_t2 + grad_e2 * (v_e2 * v_dt * t2_scale);
            grad_damping = grad_damping + grad_b_factor * v_dt;
            grad_duration_train[event] += lane_sum(
                -(grad_e1 * (v_rate1 * v_e1)) - grad_e2 * (v_rate2 * v_e2)
            );
        }

        // A real-subspace run is a single-shim one, so each parameter's row is
        // its own index; the four the representation divides out are left
        // alone, which is what the caller promised not to read.
        float row_t1[REAL_LANES], row_t2[REAL_LANES], row_m0[REAL_LANES];
        float row_b1[REAL_LANES], row_inversion[REAL_LANES], row_damping[REAL_LANES];
        lane_store(row_t1, grad_t1);
        lane_store(row_t2, grad_t2);
        lane_store(row_m0, grad_m0);
        lane_store(row_b1, grad_b1);
        lane_store(row_inversion, grad_inversion);
        lane_store(row_damping, grad_damping);
        for (std::int64_t lane = 0; lane < view.active; ++lane) {
            const std::int64_t atom = view.atom_begin + lane;
            grad_tissue_local[0 * atoms + atom] += row_t1[lane];
            grad_tissue_local[1 * atoms + atom] += row_t2[lane];
            grad_tissue_local[2 * atoms + atom] += row_m0[lane];
            grad_tissue_local[3 * atoms + atom] += row_b1[lane];
            grad_tissue_local[6 * atoms + atom] += row_inversion[lane];
            grad_tissue_local[7 * atoms + atom] += row_damping[lane];
        }
    }
}


struct VjpJvpBuffers {
    Buffers primal;
    // tangent directions for the ten differentiable inputs
    const float* dot_t1;
    const float* dot_t2;
    const float* dot_m0;
    const float* dot_b1;
    const float* dot_b1_phase;
    const float* dot_b0;
    const float* dot_inversion_efficiency;
    const float* dot_diffusion;
    const float* dot_velocity;
    // The bound pool's three properties. A single-pool run has no direction to
    // follow along them, and the kernel it selects does not read them.
    const float* dot_bound_fraction;
    const float* dot_exchange_rate;
    const float* dot_t1_bound;
    // The chemically exchanging pool's five. Read only by the kernels that
    // carry it; a run declaring the semisolid pool instead leaves them alone.
    const float* dot_pool_b_fraction;
    const float* dot_pool_b_exchange;
    const float* dot_t1_pool_b;
    const float* dot_t2_pool_b;
    const float* dot_pool_b_shift;
    const float* dot_duration;
    const float* dot_flip;
    const float* dot_phase;
    const float* grad_output_real;
    const float* grad_output_imag;
    // value part -> gradient w.r.t. the tangent inputs
    float* grad_dot_t1;
    float* grad_dot_t2;
    float* grad_dot_m0;
    float* grad_dot_b1;
    float* grad_dot_b1_phase;
    float* grad_dot_b0;
    float* grad_dot_inversion_efficiency;
    float* grad_dot_diffusion;
    float* grad_dot_velocity;
    float* grad_dot_bound_fraction;
    float* grad_dot_exchange_rate;
    float* grad_dot_t1_bound;
    // The chemically exchanging pool's five. Read only by the kernels that
    // carry it; a run declaring the semisolid pool instead leaves them alone.
    float* grad_dot_pool_b_fraction;
    float* grad_dot_pool_b_exchange;
    float* grad_dot_t1_pool_b;
    float* grad_dot_t2_pool_b;
    float* grad_dot_pool_b_shift;
    float* grad_dot_duration;
    float* grad_dot_flip;
    float* grad_dot_phase;
    // The cotangent on the per-voxel rotations, in the two planes every other
    // gradient here comes back in.
    // A direction along the per-voxel rotations, and the cotangent that comes
    // back on them in the two planes every other gradient here uses.
    const float* dynamic;
    float* grad_dot_dynamic;
    float* grad_dynamic;
    // A direction along a tabulated run's table, and its cotangent in the same
    // two planes, one table per (train, atom) in work order.
    const float* pool_values;
    float* grad_dot_pool;
    float* grad_pool;
    // tangent part -> gradient w.r.t. the primal inputs
    float* grad_t1;
    float* grad_t2;
    float* grad_m0;
    float* grad_b1;
    float* grad_b1_phase;
    float* grad_b0;
    float* grad_inversion_efficiency;
    float* grad_diffusion;
    float* grad_velocity;
    float* grad_bound_fraction;
    float* grad_exchange_rate;
    float* grad_t1_bound;
    // The chemically exchanging pool's five. Read only by the kernels that
    // carry it; a run declaring the semisolid pool instead leaves them alone.
    float* grad_pool_b_fraction;
    float* grad_pool_b_exchange;
    float* grad_t1_pool_b;
    float* grad_t2_pool_b;
    float* grad_pool_b_shift;
    float* grad_duration;
    float* grad_flip;
    float* grad_phase;
};

inline void rotate_dual(
    DualState& fplus,
    DualState& fminus,
    DualState& longitudinal,
    const DualFloat alpha,
    const DualFloat phi
) {
    const DualFloat cosine = dual_cos(alpha);
    const DualFloat sine = dual_sin(alpha);
    const DualComplex phase_one = dual_polar(phi);
    const DualComplex phase_two = phase_one * phase_one;
    const DualComplex phase_conj = conjugate(phase_one);
    const DualFloat half(DualFloat{0.5F, 0.0F});
    const DualFloat one(DualFloat{1.0F, 0.0F});
    const DualComplex t00 = to_complex(half * (one + cosine));
    const DualComplex t01 = to_complex(half * (one - cosine)) * phase_two;
    const DualComplex t02 =
        DualComplex{Complex(0.0F, -sine.value), Complex(0.0F, -sine.tangent)}
        * phase_one;
    const DualComplex t10 = conjugate(t01);
    const DualComplex t11 = t00;
    const DualComplex t12 =
        DualComplex{Complex(0.0F, sine.value), Complex(0.0F, sine.tangent)}
        * phase_conj;
    const DualFloat half_sine = half * sine;
    const DualComplex t20 =
        DualComplex{
            Complex(0.0F, -half_sine.value), Complex(0.0F, -half_sine.tangent)
        }
        * phase_conj;
    const DualComplex t21 =
        DualComplex{
            Complex(0.0F, half_sine.value), Complex(0.0F, half_sine.tangent)
        }
        * phase_one;
    const DualComplex t22 = to_complex(cosine);

    for (std::size_t state = 0; state < fplus.size(); ++state) {
        const DualComplex x0 = fplus[state];
        const DualComplex x1 = fminus[state];
        const DualComplex x2 = longitudinal[state];
        fplus[state] = t00 * x0 + t01 * x1 + t02 * x2;
        fminus[state] = t10 * x0 + t11 * x1 + t12 * x2;
        longitudinal[state] = t20 * x0 + t21 * x1 + t22 * x2;
    }
}

inline void rotate_adjoint_dual(
    const DualState& fplus_in,
    const DualState& fminus_in,
    const DualState& longitudinal_in,
    DualState& fplus_bar,
    DualState& fminus_bar,
    DualState& longitudinal_bar,
    const DualFloat alpha,
    const DualFloat phi,
    DualFloat& grad_alpha,
    DualFloat& grad_phi
) {
    const DualFloat cosine = dual_cos(alpha);
    const DualFloat sine = dual_sin(alpha);
    const DualComplex phase_one = dual_polar(phi);
    const DualComplex phase_two = phase_one * phase_one;
    const DualComplex phase_conj = conjugate(phase_one);
    const DualFloat half{0.5F, 0.0F};
    const DualFloat one{1.0F, 0.0F};

    auto imaginary_of = [](const DualFloat value) {
        return DualComplex{
            Complex(0.0F, value.value), Complex(0.0F, value.tangent)
        };
    };

    const DualComplex t00 = to_complex(half * (one + cosine));
    const DualComplex t01 = to_complex(half * (one - cosine)) * phase_two;
    const DualComplex t02 = imaginary_of(DualFloat{0.0F, 0.0F} - sine) * phase_one;
    const DualComplex t10 = conjugate(t01);
    const DualComplex t11 = t00;
    const DualComplex t12 = imaginary_of(sine) * phase_conj;
    const DualComplex t20 =
        imaginary_of(DualFloat{0.0F, 0.0F} - (half * sine)) * phase_conj;
    const DualComplex t21 = imaginary_of(half * sine) * phase_one;
    const DualComplex t22 = to_complex(cosine);

    const DualComplex d00 = to_complex(DualFloat{0.0F, 0.0F} - (half * sine));
    const DualComplex d01 = to_complex(half * sine) * phase_two;
    const DualComplex d02 =
        imaginary_of(DualFloat{0.0F, 0.0F} - cosine) * phase_one;
    const DualComplex d10 = conjugate(d01);
    const DualComplex d11 = d00;
    const DualComplex d12 = imaginary_of(cosine) * phase_conj;
    const DualComplex d20 =
        imaginary_of(DualFloat{0.0F, 0.0F} - (half * cosine)) * phase_conj;
    const DualComplex d21 = imaginary_of(half * cosine) * phase_one;
    const DualComplex d22 = to_complex(DualFloat{0.0F, 0.0F} - sine);

    const Complex imaginary(0.0F, 1.0F);
    DualFloat alpha_sum{0.0F, 0.0F};
    DualFloat phi_sum{0.0F, 0.0F};
    for (std::size_t state = 0; state < fplus_bar.size(); ++state) {
        const DualComplex a0 = fplus_bar[state];
        const DualComplex a1 = fminus_bar[state];
        const DualComplex a2 = longitudinal_bar[state];
        const DualComplex x0 = fplus_in[state];
        const DualComplex x1 = fminus_in[state];
        const DualComplex x2 = longitudinal_in[state];

        alpha_sum = alpha_sum
            + real_part(
                conjugate(a0) * (d00 * x0 + d01 * x1 + d02 * x2)
                + conjugate(a1) * (d10 * x0 + d11 * x1 + d12 * x2)
                + conjugate(a2) * (d20 * x0 + d21 * x1 + d22 * x2)
            );
        const DualComplex two{Complex(2.0F, 0.0F), Complex{}};
        phi_sum = phi_sum
            + real_part(
                conjugate(a0) * (imaginary * (two * (t01 * x1) + t02 * x2))
                + conjugate(a1)
                    * (imaginary
                       * (DualComplex{Complex(-2.0F, 0.0F), Complex{}} * (t10 * x0)
                          + DualComplex{Complex(-1.0F, 0.0F), Complex{}} * (t12 * x2)))
                + conjugate(a2)
                    * (imaginary
                       * (DualComplex{Complex(-1.0F, 0.0F), Complex{}} * (t20 * x0)
                          + t21 * x1))
            );

        fplus_bar[state] =
            conjugate(t00) * a0 + conjugate(t10) * a1 + conjugate(t20) * a2;
        fminus_bar[state] =
            conjugate(t01) * a0 + conjugate(t11) * a1 + conjugate(t21) * a2;
        longitudinal_bar[state] =
            conjugate(t02) * a0 + conjugate(t12) * a1 + conjugate(t22) * a2;
    }
    grad_alpha = grad_alpha + alpha_sum;
    grad_phi = grad_phi + phi_sum;
}

// The table read carrying a tangent in the flip angle. The pair's tangent is
// the stored slope times it, and the slope's own tangent is the Hermite
// segment's second derivative -- which its four coefficients give exactly,
// since the segment is a cubic.
inline void profile_pair_slope_dual(
    const Buffers& buffers,
    const std::int64_t row,
    const DualFloat theta,
    DualComplex& a,
    DualComplex& b,
    DualComplex& slope_a,
    DualComplex& slope_b
) {
    const float last = static_cast<float>(buffers.profile_bins - 1);
    const float step = buffers.profile_step;
    const float scaled = std::min(std::max(theta.value / step, 0.0F), last);
    const float lower = std::min(std::floor(scaled), last - 1.0F);
    const float u = scaled - lower;
    const float* const near = buffers.profile
        + (row * buffers.profile_bins + static_cast<std::int64_t>(lower))
            * PROFILE_STRIDE;
    const float* const far = near + PROFILE_STRIDE;

    const float u2 = u * u;
    const float u3 = u2 * u;
    const float h00 = 2.0F * u3 - 3.0F * u2 + 1.0F;
    const float h10 = (u3 - 2.0F * u2 + u) * step;
    const float h01 = -2.0F * u3 + 3.0F * u2;
    const float h11 = (u3 - u2) * step;
    const float g00 = (6.0F * u2 - 6.0F * u) / step;
    const float g10 = 3.0F * u2 - 4.0F * u + 1.0F;
    const float g01 = (-6.0F * u2 + 6.0F * u) / step;
    const float g11 = 3.0F * u2 - 2.0F * u;
    const float c00 = (12.0F * u - 6.0F) / (step * step);
    const float c10 = (6.0F * u - 4.0F) / step;
    const float c01 = (6.0F - 12.0F * u) / (step * step);
    const float c11 = (6.0F * u - 2.0F) / step;

    float value[4];
    float slope[4];
    float curve[4];
    for (std::size_t part = 0; part < 4; ++part) {
        value[part] = h00 * near[part] + h10 * near[part + 4]
            + h01 * far[part] + h11 * far[part + 4];
        slope[part] = g00 * near[part] + g10 * near[part + 4]
            + g01 * far[part] + g11 * far[part + 4];
        curve[part] = c00 * near[part] + c10 * near[part + 4]
            + c01 * far[part] + c11 * far[part + 4];
    }
    a = DualComplex{
        Complex(value[0], value[1]), theta.tangent * Complex(slope[0], slope[1])
    };
    b = DualComplex{
        Complex(value[2], value[3]), theta.tangent * Complex(slope[2], slope[3])
    };
    slope_a = DualComplex{
        Complex(slope[0], slope[1]), theta.tangent * Complex(curve[0], curve[1])
    };
    slope_b = DualComplex{
        Complex(slope[2], slope[3]), theta.tangent * Complex(curve[2], curve[3])
    };
}

// The spinor rotation's adjoint on dual numbers: the same closed form as
// `rotate_adjoint_spinor`, differentiated once more by the arithmetic itself.
inline void rotate_adjoint_spinor_dual(
    const DualState& fplus_in,
    const DualState& fminus_in,
    const DualState& longitudinal_in,
    DualState& fplus_bar,
    DualState& fminus_bar,
    DualState& longitudinal_bar,
    const DualComplex a,
    const DualComplex b,
    DualComplex& grad_a,
    DualComplex& grad_b
) {
    const DualComplex conj_a = conjugate(a);
    const DualComplex conj_b = conjugate(b);
    const DualComplex two{Complex(2.0F, 0.0F), Complex{}};
    const DualComplex minus_two{Complex(-2.0F, 0.0F), Complex{}};
    const DualComplex minus_one{Complex(-1.0F, 0.0F), Complex{}};

    const DualComplex t00 = conj_a * conj_a;
    const DualComplex t01 = minus_one * (conj_b * conj_b);
    const DualComplex t02 = minus_two * conjugate(a * b);
    const DualComplex t10 = minus_one * (b * b);
    const DualComplex t11 = a * a;
    const DualComplex t12 = minus_two * (a * b);
    const DualComplex t20 = conj_a * b;
    const DualComplex t21 = a * conj_b;
    const DualComplex t22 = (a * conj_a) - (b * conj_b);

    DualComplex m[3][3]{};
    for (std::size_t state = 0; state < fplus_bar.size(); ++state) {
        const DualComplex a0 = fplus_bar[state];
        const DualComplex a1 = fminus_bar[state];
        const DualComplex a2 = longitudinal_bar[state];
        const DualComplex x0 = fplus_in[state];
        const DualComplex x1 = fminus_in[state];
        const DualComplex x2 = longitudinal_in[state];

        const DualComplex s0 = conjugate(a0);
        const DualComplex s1 = conjugate(a1);
        const DualComplex s2 = conjugate(a2);
        m[0][0] = m[0][0] + s0 * x0;
        m[0][1] = m[0][1] + s0 * x1;
        m[0][2] = m[0][2] + s0 * x2;
        m[1][0] = m[1][0] + s1 * x0;
        m[1][1] = m[1][1] + s1 * x1;
        m[1][2] = m[1][2] + s1 * x2;
        m[2][0] = m[2][0] + s2 * x0;
        m[2][1] = m[2][1] + s2 * x1;
        m[2][2] = m[2][2] + s2 * x2;

        fplus_bar[state] =
            conjugate(t00) * a0 + conjugate(t10) * a1 + conjugate(t20) * a2;
        fminus_bar[state] =
            conjugate(t01) * a0 + conjugate(t11) * a1 + conjugate(t21) * a2;
        longitudinal_bar[state] =
            conjugate(t02) * a0 + conjugate(t12) * a1 + conjugate(t22) * a2;
    }

    const DualComplex holding_conj_a = two * (a * m[1][1])
        + minus_two * (b * m[1][2]) + conj_b * m[2][1] + conj_a * m[2][2];
    const DualComplex holding_a = two * (conj_a * m[0][0])
        + minus_two * (conj_b * m[0][2]) + b * m[2][0] + a * m[2][2];
    const DualComplex holding_conj_b = minus_two * (b * m[1][0])
        + minus_two * (a * m[1][2]) + conj_a * m[2][0]
        + minus_one * (conj_b * m[2][2]);
    const DualComplex holding_b = minus_two * (conj_b * m[0][1])
        + minus_two * (conj_a * m[0][2]) + a * m[2][1] + minus_one * (b * m[2][2]);
    grad_a = grad_a + conjugate(holding_conj_a) + holding_a;
    grad_b = grad_b + conjugate(holding_conj_b) + holding_b;
}

inline DualFloat dual_inverse_square(const DualFloat a) {
    const float inverse = 1.0F / (a.value * a.value);
    return {inverse, -2.0F * a.tangent * inverse / a.value};
}

inline void shift_real_dual(
    std::vector<DualFloat>& plus,
    std::vector<DualFloat>& minus,
    const std::size_t states
) {
    // No states is impossible -- every entry point rejects a state count
    // below one -- but not provable here, and the loops below count down
    // from ``states - 1``.
    if (states == 0) {
        return;
    }
    for (std::size_t state = 0; state + 1 < states; ++state) {
        minus[state] = minus[state + 1];
    }
    minus[states - 1] = DualFloat{0.0F, 0.0F};
    for (std::size_t state = states - 1; state > 0; --state) {
        plus[state] = plus[state - 1];
    }
    plus[0] = DualFloat{0.0F, 0.0F} - minus[0];
}

// Transpose of shift_real_dual. The a0 = -b0 coupling sends the incoming plus
// adjoint back onto minus, at the index the minus shift moves it to.
inline void shift_real_dual_adjoint(
    std::vector<DualFloat>& plus_bar,
    std::vector<DualFloat>& minus_bar,
    const std::size_t states
) {
    // No states is impossible -- every entry point rejects a state count
    // below one -- but not provable here, and the loops below count down
    // from ``states - 1``.
    if (states == 0) {
        return;
    }
    const DualFloat carry = DualFloat{0.0F, 0.0F} - plus_bar[0];
    for (std::size_t state = 0; state + 1 < states; ++state) {
        plus_bar[state] = plus_bar[state + 1];
    }
    plus_bar[states - 1] = DualFloat{0.0F, 0.0F};
    for (std::size_t state = states - 1; state > 0; --state) {
        minus_bar[state] = minus_bar[state - 1];
    }
    minus_bar[0] = DualFloat{0.0F, 0.0F};
    if (states > 1) {
        minus_bar[1] = minus_bar[1] + carry;
    }
}

// Forward-over-reverse through the real subspace.
//
// The forward is a chain of real linear maps, so each adjoint is a transpose
// rather than a conjugate transpose, and the state carries three reals instead
// of three complex numbers. Everything is dual-valued, giving the derivative of
// the forward-mode output with respect to the primal inputs.
//
// The RF phase does not appear: it divides out of the representation, so this
// kernel leaves grad_phase untouched. Perturbing a single pulse's phase is
// exactly the direction that leaves the subspace, so callers must not ask for
// that gradient.
void simulate_real_vjp_jvp_range(
    const VjpJvpBuffers& buffers,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count,
    DualFloat* grad_flip_local,
    DualFloat* grad_phase_local,
    DualFloat* grad_duration_local,
    DualFloat* grad_tissue_local
) {
    (void)grad_phase_local;
    const Buffers& primal = buffers.primal;
    const std::size_t states = static_cast<std::size_t>(state_count);
    const std::size_t stride = 3U * states;

    std::vector<DualFloat> trajectory(
        static_cast<std::size_t>(event_count) * stride
    );
    std::vector<DualFloat> plus(states), minus(states), longitudinal(states);
    std::vector<DualFloat> plus_bar(states), minus_bar(states), longitudinal_bar(states);
    std::vector<DualFloat> plus_stage(states), minus_stage(states), longitudinal_stage(states);
    Damping<DualFloat> damping(states);

    for (std::int64_t work = work_begin; work < work_end; ++work) {
        const TrainView view = train_view(primal, work, event_count, output_count);
        const std::int64_t atom = view.atom;
        const float* const dot_duration = buffers.dot_duration + view.event_base;
        const float* const dot_flip = buffers.dot_flip + view.event_base;
        DualFloat* const grad_flip_train = grad_flip_local + view.event_base;
        DualFloat* const grad_duration_train = grad_duration_local + view.event_base;

        const DualFloat t1{primal.t1[atom], buffers.dot_t1[atom]};
        const DualFloat t2{primal.t2[atom], buffers.dot_t2[atom]};
        const DualFloat m0 = held(primal.m0, buffers.dot_m0, atom, primal.density);
        const DualFloat b1 = held(primal.b1, buffers.dot_b1, atom, primal.transmit);
        const DualFloat inversion{
            primal.inverting ? primal.inversion_efficiency[atom] : 1.0F,
            primal.inverting ? buffers.dot_inversion_efficiency[atom] : 0.0F
        };
        const DualFloat r1{
            1000.0F / t1.value, -1000.0F * t1.tangent / (t1.value * t1.value)
        };
        const DualFloat r2{
            1000.0F / t2.value, -1000.0F * t2.tangent / (t2.value * t2.value)
        };
        const DualFloat damping_rate = held_rate(
            primal.diffusion, buffers.dot_diffusion, atom, primal.diffusing
        );

        std::fill(plus.begin(), plus.end(), DualFloat{0.0F, 0.0F});
        std::fill(minus.begin(), minus.end(), DualFloat{0.0F, 0.0F});
        std::fill(longitudinal.begin(), longitudinal.end(), DualFloat{0.0F, 0.0F});
        longitudinal[0] = DualFloat{1.0F, 0.0F};

        // ---- forward, recording the state entering each event ----
        for (std::int64_t event = 0; event < event_count; ++event) {
            DualFloat* slot = trajectory.data()
                + static_cast<std::size_t>(event) * stride;
            std::copy(plus.begin(), plus.end(), slot);
            std::copy(minus.begin(), minus.end(), slot + states);
            std::copy(longitudinal.begin(), longitudinal.end(), slot + 2U * states);

            const DualFloat dt{view.duration[event], dot_duration[event]};
            damping.set(damping_rate, dt);
            const DualFloat e1 = dual_exp(DualFloat{0.0F, 0.0F} - r1 * dt);
            const DualFloat e2 = dual_exp(DualFloat{0.0F, 0.0F} - r2 * dt);
            for (std::size_t state = 0; state < states; ++state) {
                plus[state] = plus[state] * e2 * damping.transverse[state];
                minus[state] = minus[state] * e2 * damping.transverse[state];
                longitudinal[state] =
                    longitudinal[state] * e1 * damping.longitudinal[state];
            }
            longitudinal[0] = longitudinal[0] + (DualFloat{1.0F, 0.0F} - e1);

            const std::uint8_t action = primal.action[event];
            if ((action & PRE_SHIFT) != 0) {
                shift_real_dual(plus, minus, states);
            }
            if (primal.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    const DualFloat efficiency = DualFloat{0.0F, 0.0F} - inversion;
                    for (std::size_t state = 0; state < states; ++state) {
                        longitudinal[state] = longitudinal[state] * efficiency;
                    }
                } else {
                    const DualFloat alpha =
                        DualFloat{view.flip[event], dot_flip[event]} * b1;
                    const DualFloat cosine{
                        std::cos(alpha.value), -std::sin(alpha.value) * alpha.tangent
                    };
                    const DualFloat sine{
                        std::sin(alpha.value), std::cos(alpha.value) * alpha.tangent
                    };
                    const DualFloat cosine_half_sq =
                        0.5F * (DualFloat{1.0F, 0.0F} + cosine);
                    const DualFloat sine_half_sq =
                        0.5F * (DualFloat{1.0F, 0.0F} - cosine);
                    const DualFloat half_sine = 0.5F * sine;
                    for (std::size_t state = 0; state < states; ++state) {
                        const DualFloat p = plus[state];
                        const DualFloat m = minus[state];
                        const DualFloat z = longitudinal[state];
                        plus[state] = cosine_half_sq * p + sine_half_sq * m
                            - sine * z;
                        minus[state] = sine_half_sq * p + cosine_half_sq * m
                            + sine * z;
                        longitudinal[state] = half_sine * p - half_sine * m
                            + cosine * z;
                    }
                }
            }
            if ((action & POST_SHIFT) != 0) {
                shift_real_dual(plus, minus, states);
            }
            if ((action & SPOIL_AFTER) != 0) {
                std::fill(plus.begin(), plus.end(), DualFloat{0.0F, 0.0F});
                std::fill(minus.begin(), minus.end(), DualFloat{0.0F, 0.0F});
            } else if ((action & SHIFT_AFTER) != 0) {
                shift_real_dual(plus, minus, states);
            }
        }

        // ---- reverse ----
        std::fill(plus_bar.begin(), plus_bar.end(), DualFloat{0.0F, 0.0F});
        std::fill(minus_bar.begin(), minus_bar.end(), DualFloat{0.0F, 0.0F});
        std::fill(
            longitudinal_bar.begin(), longitudinal_bar.end(), DualFloat{0.0F, 0.0F}
        );
        DualFloat grad_t1{0.0F, 0.0F};
        DualFloat grad_t2{0.0F, 0.0F};
        DualFloat grad_m0{0.0F, 0.0F};
        DualFloat grad_b1{0.0F, 0.0F};
        DualFloat grad_inversion{0.0F, 0.0F};
        DualFloat grad_damping{0.0F, 0.0F};

        for (std::int64_t event = event_count - 1; event >= 0; --event) {
            const DualFloat* slot = trajectory.data()
                + static_cast<std::size_t>(event) * stride;
            const std::uint8_t action = primal.action[event];
            const DualFloat dt{view.duration[event], dot_duration[event]};
            damping.set(damping_rate, dt);
            const DualFloat e1 = dual_exp(DualFloat{0.0F, 0.0F} - r1 * dt);
            const DualFloat e2 = dual_exp(DualFloat{0.0F, 0.0F} - r2 * dt);

            // Replay the intra-event stages from the recorded entry state.
            for (std::size_t state = 0; state < states; ++state) {
                plus_stage[state] = slot[state] * e2 * damping.transverse[state];
                minus_stage[state] =
                    slot[states + state] * e2 * damping.transverse[state];
                longitudinal_stage[state] =
                    slot[2U * states + state] * e1 * damping.longitudinal[state];
            }
            longitudinal_stage[0] =
                longitudinal_stage[0] + (DualFloat{1.0F, 0.0F} - e1);
            if ((action & PRE_SHIFT) != 0) {
                shift_real_dual(plus_stage, minus_stage, states);
            }
            // plus_stage now holds the state entering the RF operator.

            // Undo the trailing spoil/shift.
            if ((action & SPOIL_AFTER) != 0) {
                std::fill(plus_bar.begin(), plus_bar.end(), DualFloat{0.0F, 0.0F});
                std::fill(minus_bar.begin(), minus_bar.end(), DualFloat{0.0F, 0.0F});
            } else if ((action & SHIFT_AFTER) != 0) {
                shift_real_dual_adjoint(plus_bar, minus_bar, states);
            }
            if ((action & POST_SHIFT) != 0) {
                shift_real_dual_adjoint(plus_bar, minus_bar, states);
            }

            if (primal.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    const DualFloat efficiency = DualFloat{0.0F, 0.0F} - inversion;
                    for (std::size_t state = 0; state < states; ++state) {
                        grad_inversion = grad_inversion
                            - longitudinal_bar[state] * longitudinal_stage[state];
                        longitudinal_bar[state] = longitudinal_bar[state] * efficiency;
                    }
                } else {
                    const DualFloat flip{view.flip[event], dot_flip[event]};
                    const DualFloat alpha = flip * b1;
                    const DualFloat cosine{
                        std::cos(alpha.value), -std::sin(alpha.value) * alpha.tangent
                    };
                    const DualFloat sine{
                        std::sin(alpha.value), std::cos(alpha.value) * alpha.tangent
                    };
                    const DualFloat cosine_half_sq =
                        0.5F * (DualFloat{1.0F, 0.0F} + cosine);
                    const DualFloat sine_half_sq =
                        0.5F * (DualFloat{1.0F, 0.0F} - cosine);
                    const DualFloat half_sine = 0.5F * sine;
                    DualFloat grad_alpha{0.0F, 0.0F};
                    for (std::size_t state = 0; state < states; ++state) {
                        const DualFloat p = plus_stage[state];
                        const DualFloat m = minus_stage[state];
                        const DualFloat z = longitudinal_stage[state];
                        const DualFloat pb = plus_bar[state];
                        const DualFloat mb = minus_bar[state];
                        const DualFloat zb = longitudinal_bar[state];
                        // d/dalpha of each output row, contracted with the adjoint.
                        grad_alpha = grad_alpha
                            + pb * (half_sine * m - half_sine * p - cosine * z)
                            + mb * (half_sine * p - half_sine * m + cosine * z)
                            + zb * (0.5F * cosine * p - 0.5F * cosine * m - sine * z);
                        // Transpose of the rotation.
                        plus_bar[state] = cosine_half_sq * pb + sine_half_sq * mb
                            + half_sine * zb;
                        minus_bar[state] = sine_half_sq * pb + cosine_half_sq * mb
                            - half_sine * zb;
                        longitudinal_bar[state] = (DualFloat{0.0F, 0.0F} - sine) * pb
                            + sine * mb + cosine * zb;
                    }
                    grad_flip_train[event] = grad_flip_train[event] + grad_alpha * b1;
                    grad_b1 = grad_b1 + grad_alpha * flip;
                }
            } else if (primal.kind[event] == 2 && (action & RECORD) != 0) {
                // The sample is i * m0 * plus[0]; only the imaginary seed acts.
                const std::int64_t index =
                    view.output_base + primal.output_index[event];
                const DualFloat seed{buffers.grad_output_imag[index], 0.0F};
                grad_m0 = grad_m0 + seed * plus_stage[0];
                plus_bar[0] = plus_bar[0] + seed * m0;
            }

            if ((action & PRE_SHIFT) != 0) {
                shift_real_dual_adjoint(plus_bar, minus_bar, states);
            }

            DualFloat grad_e1{0.0F, 0.0F};
            DualFloat grad_e2{0.0F, 0.0F};
            DualFloat grad_b_factor{0.0F, 0.0F};
            grad_e1 = grad_e1 - longitudinal_bar[0];
            for (std::size_t state = 0; state < states; ++state) {
                const DualFloat damp_transverse = damping.transverse[state];
                const DualFloat damp_longitudinal = damping.longitudinal[state];
                const DualFloat plus_term =
                    plus_bar[state] * slot[state] * damp_transverse;
                const DualFloat minus_term =
                    minus_bar[state] * slot[states + state] * damp_transverse;
                const DualFloat longitudinal_term = longitudinal_bar[state]
                    * slot[2U * states + state] * damp_longitudinal;
                grad_e2 = grad_e2 + plus_term + minus_term;
                grad_e1 = grad_e1 + longitudinal_term;
                grad_b_factor = grad_b_factor
                    - transverse_weight(state) * (e2 * (plus_term + minus_term))
                    - longitudinal_weight(state) * (e1 * longitudinal_term);
                plus_bar[state] = plus_bar[state] * e2 * damp_transverse;
                minus_bar[state] = minus_bar[state] * e2 * damp_transverse;
                longitudinal_bar[state] =
                    longitudinal_bar[state] * e1 * damp_longitudinal;
            }
            const DualFloat scale1 =
                e1 * dt * (1000.0F * dual_inverse_square(t1));
            const DualFloat scale2 =
                e2 * dt * (1000.0F * dual_inverse_square(t2));
            grad_t1 = grad_t1 + grad_e1 * scale1;
            grad_t2 = grad_t2 + grad_e2 * scale2;
            grad_damping = grad_damping + grad_b_factor * dt;
            grad_duration_train[event] = grad_duration_train[event]
                - grad_e1 * (r1 * e1) - grad_e2 * (r2 * e2)
                + grad_b_factor * damping_rate;
        }

        const std::int64_t atoms = primal.atom_count;
        const DualFloat contributions[TISSUE_COUNT] = {
            grad_t1, grad_t2, grad_m0, grad_b1, DualFloat{0.0F, 0.0F},
            DualFloat{0.0F, 0.0F}, grad_inversion, grad_damping,
            DualFloat{0.0F, 0.0F},
            // The second pools' properties, left at zero: neither pool is
            // inside the real subspace this kernel stands for, so the dispatch
            // never sends one here.
            DualFloat{0.0F, 0.0F}, DualFloat{0.0F, 0.0F},
            DualFloat{0.0F, 0.0F}, DualFloat{0.0F, 0.0F},
            DualFloat{0.0F, 0.0F}, DualFloat{0.0F, 0.0F},
            DualFloat{0.0F, 0.0F}, DualFloat{0.0F, 0.0F},
        };
        for (std::size_t parameter = 0; parameter < TISSUE_COUNT; ++parameter) {
            DualFloat& target = grad_tissue_local[parameter * atoms + atom];
            target = target + contributions[parameter];
        }
    }
}

// Lane-parallel dual: the value and tangent planes each hold one train per lane.
struct DualLane {
    LaneVector value;
    LaneVector tangent;
};

inline DualLane operator+(const DualLane left, const DualLane right) {
    return DualLane{left.value + right.value, left.tangent + right.tangent};
}

inline DualLane operator-(const DualLane left, const DualLane right) {
    return DualLane{left.value - right.value, left.tangent - right.tangent};
}

inline DualLane operator*(const DualLane left, const DualLane right) {
    return DualLane{
        left.value * right.value,
        left.value * right.tangent + left.tangent * right.value,
    };
}

inline DualLane operator-(const DualLane value) {
    return DualLane{-value.value, -value.tangent};
}

inline DualLane operator*(const float left, const DualLane right) {
    const LaneVector scale = lane_splat(left);
    return DualLane{scale * right.value, scale * right.tangent};
}

inline DualLane dual_lane_splat(const DualFloat value) {
    return DualLane{lane_splat(value.value), lane_splat(value.tangent)};
}

inline DualLane dual_lane_zero() {
    const LaneVector zero = lane_splat(0.0F);
    return DualLane{zero, zero};
}

inline DualLane dual_lane_exp(const DualLane argument) {
    float buffer[REAL_LANES];
    lane_store(buffer, argument.value);
    for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
        buffer[lane] = std::exp(buffer[lane]);
    }
    const LaneVector value = lane_load(buffer);
    return DualLane{value, argument.tangent * value};
}

struct DualLaneTrig {
    DualLane cosine;
    DualLane sine;
};

inline DualLaneTrig dual_lane_trig(const DualLane angle) {
    float value[REAL_LANES], cosine[REAL_LANES], sine[REAL_LANES];
    lane_store(value, angle.value);
    for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
        cosine[lane] = std::cos(value[lane]);
        sine[lane] = std::sin(value[lane]);
    }
    const LaneVector c = lane_load(cosine);
    const LaneVector s = lane_load(sine);
    return DualLaneTrig{
        DualLane{c, -(s * angle.tangent)},
        DualLane{s, c * angle.tangent},
    };
}

// Inactive lanes repeat the block's first train, so a reduction that crosses the
// lane axis must stop at ``active`` or it would count that train twice.
inline float lane_sum(const LaneVector value, const std::int64_t active) {
    float buffer[REAL_LANES];
    lane_store(buffer, value);
    float total = 0.0F;
    for (std::int64_t lane = 0; lane < active; ++lane) {
        total += buffer[lane];
    }
    return total;
}

inline void shift_real_dual_lanes(
    DualLane* plus, DualLane* minus, const std::size_t states
) {
    // No states is impossible -- every entry point rejects a state count
    // below one -- but not provable here, and the loops below count down
    // from ``states - 1``.
    if (states == 0) {
        return;
    }
    for (std::size_t state = 0; state + 1 < states; ++state) {
        minus[state] = minus[state + 1];
    }
    minus[states - 1] = dual_lane_zero();
    for (std::size_t state = states - 1; state > 0; --state) {
        plus[state] = plus[state - 1];
    }
    plus[0] = -minus[0];
}

inline void shift_real_dual_lanes_adjoint(
    DualLane* plus_bar, DualLane* minus_bar, const std::size_t states
) {
    // No states is impossible -- every entry point rejects a state count
    // below one -- but not provable here, and the loops below count down
    // from ``states - 1``.
    if (states == 0) {
        return;
    }
    const DualLane carry = -plus_bar[0];
    for (std::size_t state = 0; state + 1 < states; ++state) {
        plus_bar[state] = plus_bar[state + 1];
    }
    plus_bar[states - 1] = dual_lane_zero();
    for (std::size_t state = states - 1; state > 0; --state) {
        minus_bar[state] = minus_bar[state - 1];
    }
    minus_bar[0] = dual_lane_zero();
    if (states > 1) {
        minus_bar[1] = minus_bar[1] + carry;
    }
}

void simulate_real_vjp_jvp_lane_range(
    const VjpJvpBuffers& buffers,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count,
    DualFloat* grad_flip_local,
    DualFloat* grad_phase_local,
    DualFloat* grad_duration_local,
    DualFloat* grad_tissue_local
) {
    (void)grad_phase_local;
    const Buffers& primal = buffers.primal;
    const std::size_t states = static_cast<std::size_t>(state_count);
    const std::size_t stride = 3U * states;

    std::vector<DualLane> trajectory(
        static_cast<std::size_t>(event_count) * stride
    );
    std::vector<DualLane> plus(states), minus(states), longitudinal(states);
    std::vector<DualLane> plus_bar(states), minus_bar(states);
    std::vector<DualLane> longitudinal_bar(states);
    std::vector<DualLane> plus_stage(states), minus_stage(states);
    std::vector<DualLane> longitudinal_stage(states);

    float dt_value[REAL_LANES], dt_tangent[REAL_LANES];
    float flip_value[REAL_LANES], flip_tangent[REAL_LANES];
    float seed_value[REAL_LANES];
    float scatter_value[REAL_LANES], scatter_tangent[REAL_LANES];

    const DualLane zero = dual_lane_zero();
    const DualLane one = DualLane{lane_splat(1.0F), lane_splat(0.0F)};

    for (std::int64_t work = work_begin; work < work_end; ++work) {
        const LaneView view = lane_view(primal, work);
        const std::int64_t atom = view.atom;

        const DualFloat t1{primal.t1[atom], buffers.dot_t1[atom]};
        const DualFloat t2{primal.t2[atom], buffers.dot_t2[atom]};
        const DualLane m0 = dual_lane_splat(
            held(primal.m0, buffers.dot_m0, atom, primal.density)
        );
        const DualLane b1 = dual_lane_splat(
            held(primal.b1, buffers.dot_b1, atom, primal.transmit)
        );
        const DualLane inversion = dual_lane_splat(DualFloat{
            primal.inversion_efficiency[atom],
            buffers.dot_inversion_efficiency[atom],
        });
        const DualLane r1 = dual_lane_splat(DualFloat{
            1000.0F / t1.value, -1000.0F * t1.tangent / (t1.value * t1.value)
        });
        const DualLane r2 = dual_lane_splat(DualFloat{
            1000.0F / t2.value, -1000.0F * t2.tangent / (t2.value * t2.value)
        });
        const DualLane inverse_t1 =
            dual_lane_splat(1000.0F * dual_inverse_square(t1));
        const DualLane inverse_t2 =
            dual_lane_splat(1000.0F * dual_inverse_square(t2));

        std::fill(plus.begin(), plus.end(), zero);
        std::fill(minus.begin(), minus.end(), zero);
        std::fill(longitudinal.begin(), longitudinal.end(), zero);
        longitudinal[0] = one;

        // ---- forward, recording the state entering each event ----
        for (std::int64_t event = 0; event < event_count; ++event) {
            DualLane* slot = trajectory.data()
                + static_cast<std::size_t>(event) * stride;
            std::copy(plus.begin(), plus.end(), slot);
            std::copy(minus.begin(), minus.end(), slot + states);
            std::copy(longitudinal.begin(), longitudinal.end(), slot + 2U * states);

            gather_lanes(primal.duration, view, event, event_count, dt_value);
            gather_lanes(buffers.dot_duration, view, event, event_count, dt_tangent);
            const DualLane dt{lane_load(dt_value), lane_load(dt_tangent)};
            const DualLane e1 = dual_lane_exp(-(r1 * dt));
            const DualLane e2 = dual_lane_exp(-(r2 * dt));
            for (std::size_t state = 0; state < states; ++state) {
                plus[state] = plus[state] * e2;
                minus[state] = minus[state] * e2;
                longitudinal[state] = longitudinal[state] * e1;
            }
            longitudinal[0] = longitudinal[0] + (one - e1);

            const std::uint8_t action = primal.action[event];
            if ((action & PRE_SHIFT) != 0) {
                shift_real_dual_lanes(plus.data(), minus.data(), states);
            }
            if (primal.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    const DualLane efficiency = -inversion;
                    for (std::size_t state = 0; state < states; ++state) {
                        longitudinal[state] = longitudinal[state] * efficiency;
                    }
                } else {
                    gather_lanes(primal.flip, view, event, event_count, flip_value);
                    gather_lanes(
                        buffers.dot_flip, view, event, event_count, flip_tangent
                    );
                    const DualLane flip{
                        lane_load(flip_value), lane_load(flip_tangent)
                    };
                    const DualLaneTrig trig = dual_lane_trig(flip * b1);
                    const DualLane cosine_half_sq = 0.5F * (one + trig.cosine);
                    const DualLane sine_half_sq = 0.5F * (one - trig.cosine);
                    const DualLane half_sine = 0.5F * trig.sine;
                    for (std::size_t state = 0; state < states; ++state) {
                        const DualLane p = plus[state];
                        const DualLane m = minus[state];
                        const DualLane z = longitudinal[state];
                        plus[state] = cosine_half_sq * p + sine_half_sq * m
                            - trig.sine * z;
                        minus[state] = sine_half_sq * p + cosine_half_sq * m
                            + trig.sine * z;
                        longitudinal[state] = half_sine * p - half_sine * m
                            + trig.cosine * z;
                    }
                }
            }
            if ((action & POST_SHIFT) != 0) {
                shift_real_dual_lanes(plus.data(), minus.data(), states);
            }
            if ((action & SPOIL_AFTER) != 0) {
                std::fill(plus.begin(), plus.end(), zero);
                std::fill(minus.begin(), minus.end(), zero);
            } else if ((action & SHIFT_AFTER) != 0) {
                shift_real_dual_lanes(plus.data(), minus.data(), states);
            }
        }

        // ---- reverse ----
        std::fill(plus_bar.begin(), plus_bar.end(), zero);
        std::fill(minus_bar.begin(), minus_bar.end(), zero);
        std::fill(longitudinal_bar.begin(), longitudinal_bar.end(), zero);
        DualLane grad_t1 = zero;
        DualLane grad_t2 = zero;
        DualLane grad_m0 = zero;
        DualLane grad_b1 = zero;
        DualLane grad_inversion = zero;
        DualLane grad_damping = zero;

        for (std::int64_t event = event_count - 1; event >= 0; --event) {
            const DualLane* slot = trajectory.data()
                + static_cast<std::size_t>(event) * stride;
            const std::uint8_t action = primal.action[event];
            gather_lanes(primal.duration, view, event, event_count, dt_value);
            gather_lanes(buffers.dot_duration, view, event, event_count, dt_tangent);
            const DualLane dt{lane_load(dt_value), lane_load(dt_tangent)};
            const DualLane e1 = dual_lane_exp(-(r1 * dt));
            const DualLane e2 = dual_lane_exp(-(r2 * dt));

            // Replay the intra-event stages from the recorded entry state.
            for (std::size_t state = 0; state < states; ++state) {
                plus_stage[state] = slot[state] * e2;
                minus_stage[state] = slot[states + state] * e2;
                longitudinal_stage[state] = slot[2U * states + state] * e1;
            }
            longitudinal_stage[0] = longitudinal_stage[0] + (one - e1);
            if ((action & PRE_SHIFT) != 0) {
                shift_real_dual_lanes(plus_stage.data(), minus_stage.data(), states);
            }

            if ((action & SPOIL_AFTER) != 0) {
                std::fill(plus_bar.begin(), plus_bar.end(), zero);
                std::fill(minus_bar.begin(), minus_bar.end(), zero);
            } else if ((action & SHIFT_AFTER) != 0) {
                shift_real_dual_lanes_adjoint(
                    plus_bar.data(), minus_bar.data(), states
                );
            }
            if ((action & POST_SHIFT) != 0) {
                shift_real_dual_lanes_adjoint(
                    plus_bar.data(), minus_bar.data(), states
                );
            }

            if (primal.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    const DualLane efficiency = -inversion;
                    for (std::size_t state = 0; state < states; ++state) {
                        grad_inversion = grad_inversion
                            - longitudinal_bar[state] * longitudinal_stage[state];
                        longitudinal_bar[state] = longitudinal_bar[state] * efficiency;
                    }
                } else {
                    gather_lanes(primal.flip, view, event, event_count, flip_value);
                    gather_lanes(
                        buffers.dot_flip, view, event, event_count, flip_tangent
                    );
                    const DualLane flip{
                        lane_load(flip_value), lane_load(flip_tangent)
                    };
                    const DualLaneTrig trig = dual_lane_trig(flip * b1);
                    const DualLane cosine_half_sq = 0.5F * (one + trig.cosine);
                    const DualLane sine_half_sq = 0.5F * (one - trig.cosine);
                    const DualLane half_sine = 0.5F * trig.sine;
                    DualLane grad_alpha = zero;
                    for (std::size_t state = 0; state < states; ++state) {
                        const DualLane p = plus_stage[state];
                        const DualLane m = minus_stage[state];
                        const DualLane z = longitudinal_stage[state];
                        const DualLane pb = plus_bar[state];
                        const DualLane mb = minus_bar[state];
                        const DualLane zb = longitudinal_bar[state];
                        grad_alpha = grad_alpha
                            + pb * (half_sine * m - half_sine * p - trig.cosine * z)
                            + mb * (half_sine * p - half_sine * m + trig.cosine * z)
                            + zb * (0.5F * trig.cosine * p - 0.5F * trig.cosine * m
                                    - trig.sine * z);
                        plus_bar[state] = cosine_half_sq * pb + sine_half_sq * mb
                            + half_sine * zb;
                        minus_bar[state] = sine_half_sq * pb + cosine_half_sq * mb
                            - half_sine * zb;
                        longitudinal_bar[state] = -trig.sine * pb + trig.sine * mb
                            + trig.cosine * zb;
                    }
                    const DualLane contribution = grad_alpha * b1;
                    lane_store(scatter_value, contribution.value);
                    lane_store(scatter_tangent, contribution.tangent);
                    for (std::int64_t lane = 0; lane < view.active; ++lane) {
                        DualFloat& target = grad_flip_local
                            [(view.train_begin + lane) * event_count + event];
                        target.value += scatter_value[lane];
                        target.tangent += scatter_tangent[lane];
                    }
                    grad_b1 = grad_b1 + grad_alpha * flip;
                }
            } else if (primal.kind[event] == 2 && (action & RECORD) != 0) {
                // The sample is i * m0 * plus[0]; only the imaginary seed acts.
                const std::int64_t offset = primal.output_index[event];
                for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
                    seed_value[lane] = 0.0F;
                }
                for (std::int64_t lane = 0; lane < view.active; ++lane) {
                    const std::int64_t index =
                        ((view.train_begin + lane) * primal.atom_count + atom)
                            * output_count
                        + offset;
                    seed_value[lane] = buffers.grad_output_imag[index];
                }
                const DualLane seed{lane_load(seed_value), lane_splat(0.0F)};
                grad_m0 = grad_m0 + seed * plus_stage[0];
                plus_bar[0] = plus_bar[0] + seed * m0;
            }

            if ((action & PRE_SHIFT) != 0) {
                shift_real_dual_lanes_adjoint(
                    plus_bar.data(), minus_bar.data(), states
                );
            }

            DualLane grad_e1 = -longitudinal_bar[0];
            DualLane grad_e2 = zero;
            // The dispatch reaches this kernel only where the damping rate is
            // zero, so every per-order factor above is exactly one and the
            // b-factor's gradient is a weighted sum with no exponential in it.
            DualLane grad_b_factor = zero;
            for (std::size_t state = 0; state < states; ++state) {
                const DualLane plus_term = plus_bar[state] * slot[state];
                const DualLane minus_term =
                    minus_bar[state] * slot[states + state];
                const DualLane longitudinal_term =
                    longitudinal_bar[state] * slot[2U * states + state];
                grad_e2 = grad_e2 + plus_term + minus_term;
                grad_e1 = grad_e1 + longitudinal_term;
                grad_b_factor = grad_b_factor
                    - transverse_weight(state) * (e2 * (plus_term + minus_term))
                    - longitudinal_weight(state) * (e1 * longitudinal_term);
                plus_bar[state] = plus_bar[state] * e2;
                minus_bar[state] = minus_bar[state] * e2;
                longitudinal_bar[state] = longitudinal_bar[state] * e1;
            }
            grad_t1 = grad_t1 + grad_e1 * (e1 * dt * inverse_t1);
            grad_t2 = grad_t2 + grad_e2 * (e2 * dt * inverse_t2);
            grad_damping = grad_damping + grad_b_factor * dt;
            const DualLane grad_dt =
                -(grad_e1 * (r1 * e1)) - grad_e2 * (r2 * e2);
            lane_store(scatter_value, grad_dt.value);
            lane_store(scatter_tangent, grad_dt.tangent);
            for (std::int64_t lane = 0; lane < view.active; ++lane) {
                DualFloat& target = grad_duration_local
                    [(view.train_begin + lane) * event_count + event];
                target.value += scatter_value[lane];
                target.tangent += scatter_tangent[lane];
            }
        }

        const std::int64_t atoms = primal.atom_count;
        const DualLane contributions[TISSUE_COUNT] = {
            grad_t1, grad_t2, grad_m0, grad_b1, zero, zero, grad_inversion,
            grad_damping, zero,
            // The second pools' properties, left at zero: neither pool is
            // inside the real subspace this kernel stands for, so the dispatch
            // never sends one here.
            zero, zero, zero, zero, zero, zero, zero, zero,
        };
        for (std::size_t parameter = 0; parameter < TISSUE_COUNT; ++parameter) {
            DualFloat& target = grad_tissue_local[parameter * atoms + atom];
            target.value += lane_sum(contributions[parameter].value, view.active);
            target.tangent += lane_sum(contributions[parameter].tangent, view.active);
        }
    }
}

// One half of the transmit pair a pulse sees, with its tangent, read at the
// row the pulse's shim drives. A single-shim sequence names row zero on every
// event, which is that voxel's own field.
inline DualFloat shim_dual(
    const bool shimmed,
    const float* const value,
    const float* const tangent,
    const std::int64_t row,
    const DualFloat voxel
) {
    return shimmed ? DualFloat{value[row], tangent[row]} : voxel;
}

// Inlined into the vector clones below rather than called from them: a
// clone is a copy of the caller, so a body reached through a call is
// compiled once for the baseline instruction set and no more.
template <RfMode MODE, Pools POOLS>
TORCHSIM_ALWAYS_INLINE void simulate_vjp_jvp_range(
    const VjpJvpBuffers& buffers,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count,
    DualFloat* grad_flip_local,
    DualFloat* grad_phase_local,
    DualFloat* grad_duration_local,
    // Per-atom accumulators laid out [row][atom]; see the first-order kernel
    // for the rows and for why these cannot be written directly.
    DualFloat* grad_tissue_local
) {
    constexpr bool MT = POOLS == Pools::SEMISOLID;
    constexpr bool BM = POOLS == Pools::EXCHANGING;
    constexpr bool THREE = POOLS == Pools::THREE;
    constexpr bool TWO_POOL = MT || BM;
    constexpr bool PAIRED = BM || THREE;
    constexpr bool SATURATED = MT || THREE;
    const Buffers& primal = buffers.primal;
    const TissueLayout layout(primal.shim_count);
    const std::int64_t atoms = primal.atom_count;
    const std::size_t states = static_cast<std::size_t>(state_count);
    const std::size_t stride = (THREE ? 7U : (BM ? 6U : (MT ? 4U : 3U))) * states;

    std::vector<DualComplex> trajectory(
        static_cast<std::size_t>(event_count) * stride
    );
    DualState fplus(states);
    DualState fminus(states);
    DualState longitudinal(states);
    DualState fplus_bar(states);
    DualState fminus_bar(states);
    DualState longitudinal_bar(states);
    DualState fplus_relaxed(states);
    DualState fminus_relaxed(states);
    DualState longitudinal_relaxed(states);
    DualState fplus_shifted(states);
    DualState fminus_shifted(states);
    DualState bound((TWO_POOL || THREE) ? states : 0U);
    DualState bound_bar((TWO_POOL || THREE) ? states : 0U);
    DualState bound_relaxed((TWO_POOL || THREE) ? states : 0U);
    DualState semisolid(THREE ? states : 0U);
    DualState semisolid_bar(THREE ? states : 0U);
    DualState semisolid_relaxed(THREE ? states : 0U);
    DualState bound_plus(PAIRED ? states : 0U);
    DualState bound_minus(PAIRED ? states : 0U);
    DualState bound_plus_bar(PAIRED ? states : 0U);
    DualState bound_minus_bar(PAIRED ? states : 0U);
    DualState bound_plus_relaxed(PAIRED ? states : 0U);
    DualState bound_minus_relaxed(PAIRED ? states : 0U);
    DualState bound_plus_shifted(PAIRED ? states : 0U);
    DualState bound_minus_shifted(PAIRED ? states : 0U);
    Damping<DualFloat> damping(states);

    for (std::int64_t work = work_begin; work < work_end; ++work) {
        const TrainView view = train_view(primal, work, event_count, output_count);
        const std::int64_t atom = view.atom;
        const std::int64_t location = slice_row<MODE>(primal, atom);
        const float* const dot_duration = buffers.dot_duration + view.event_base;
        const float* const dot_flip = buffers.dot_flip + view.event_base;
        const float* const dot_phase = buffers.dot_phase + view.event_base;
        DualFloat* const grad_flip_train = grad_flip_local + view.event_base;
        DualFloat* const grad_phase_train = grad_phase_local + view.event_base;
        DualFloat* const grad_duration_train = grad_duration_local + view.event_base;
        const DualFloat t1{primal.t1[atom], buffers.dot_t1[atom]};
        const DualFloat t2{primal.t2[atom], buffers.dot_t2[atom]};
        const DualFloat m0 = held(primal.m0, buffers.dot_m0, atom, primal.density);
        const DualFloat b1 = held(primal.b1, buffers.dot_b1, atom, primal.transmit);
        const DualFloat b1_phase =
            held_rate(primal.b1_phase, buffers.dot_b1_phase, atom, primal.off_axis);
        const bool shimmed = primal.shim_count > 1;
        const DualFloat b0 =
            held_rate(primal.b0, buffers.dot_b0, atom, primal.off_axis);
        const DualFloat efficiency{
            primal.inversion_efficiency[atom],
            buffers.dot_inversion_efficiency[atom],
        };
        const DualFloat r1 = 1000.0F * dual_reciprocal(t1);
        const DualFloat r2 = 1000.0F * dual_reciprocal(t2);
        const DualFloat bound_fraction = (BM || THREE)
            ? DualFloat{
                primal.pool_b_fraction[atom],
                buffers.dot_pool_b_fraction[atom],
            }
            : (MT
                ? DualFloat{
                    primal.bound_fraction[atom], buffers.dot_bound_fraction[atom]
                }
                : DualFloat{});
        const DualFloat semisolid_fraction = SATURATED
            ? DualFloat{
                primal.bound_fraction[atom], buffers.dot_bound_fraction[atom]
            }
            : DualFloat{};
        const DualFloat exchange = (BM || THREE)
            ? DualFloat{
                primal.pool_b_exchange[atom],
                buffers.dot_pool_b_exchange[atom],
            }
            : (MT
                ? DualFloat{
                    primal.bound_exchange[atom], buffers.dot_exchange_rate[atom]
                }
                : DualFloat{});
        const DualFloat semisolid_exchange = SATURATED
            ? DualFloat{
                primal.bound_exchange[atom], buffers.dot_exchange_rate[atom]
            }
            : DualFloat{};
        const DualFloat t1_bound = (BM || THREE)
            ? DualFloat{primal.t1_pool_b[atom], buffers.dot_t1_pool_b[atom]}
            : (MT
                ? DualFloat{primal.t1_bound[atom], buffers.dot_t1_bound[atom]}
                : DualFloat{1.0F, 0.0F});
        const DualFloat t1_semisolid = SATURATED
            ? DualFloat{primal.t1_bound[atom], buffers.dot_t1_bound[atom]}
            : DualFloat{1.0F, 0.0F};
        const DualFloat r1_semisolid =
            SATURATED ? 1000.0F * dual_reciprocal(t1_semisolid) : DualFloat{};
        const DualFloat r1_bound =
            (TWO_POOL || THREE) ? 1000.0F * dual_reciprocal(t1_bound) : DualFloat{};
        const DualFloat t2_bound = PAIRED
            ? DualFloat{primal.t2_pool_b[atom], buffers.dot_t2_pool_b[atom]}
            : DualFloat{1.0F, 0.0F};
        const DualFloat r2_bound =
            PAIRED ? 1000.0F * dual_reciprocal(t2_bound) : DualFloat{};
        const DualFloat transverse_free = THREE
            ? DualFloat{1.0F, 0.0F} - bound_fraction - semisolid_fraction
            : DualFloat{1.0F, 0.0F} - bound_fraction;
        const DualFloat pool_shift = PAIRED
            ? DualFloat{
                primal.pool_b_shift[atom], buffers.dot_pool_b_shift[atom]
            }
            : DualFloat{};
        const DualFloat damping_rate = held_rate(
            primal.diffusion, buffers.dot_diffusion, atom, primal.diffusing
        );
        const float velocity = primal.moving ? primal.velocity[atom] : 0.0F;
        const float dot_velocity =
            primal.moving ? buffers.dot_velocity[atom] : 0.0F;
        const DualFloat flow_rate{
            velocity * primal.flow_scale, dot_velocity * primal.flow_scale
        };
        const DualFloat washout_rate{
            std::fabs(velocity) * primal.washout_scale,
            speed_direction(velocity) * dot_velocity * primal.washout_scale,
        };

        std::fill(fplus.begin(), fplus.end(), DualComplex{});
        std::fill(fminus.begin(), fminus.end(), DualComplex{});
        std::fill(longitudinal.begin(), longitudinal.end(), DualComplex{});
        const DualFloat held_free = THREE
            ? DualFloat{1.0F, 0.0F} - bound_fraction - semisolid_fraction
            : DualFloat{1.0F, 0.0F} - bound_fraction;
        longitudinal[0] = DualComplex{
            Complex(held_free.value, 0.0F), Complex(held_free.tangent, 0.0F)
        };
        if constexpr (THREE) {
            std::fill(semisolid.begin(), semisolid.end(), DualComplex{});
            semisolid[0] = DualComplex{
                Complex(semisolid_fraction.value, 0.0F),
                Complex(semisolid_fraction.tangent, 0.0F),
            };
        }
        if constexpr (TWO_POOL || THREE) {
            std::fill(bound.begin(), bound.end(), DualComplex{});
            bound[0] = DualComplex{
                Complex(bound_fraction.value, 0.0F),
                Complex(bound_fraction.tangent, 0.0F),
            };
        }
        if constexpr (PAIRED) {
            std::fill(bound_plus.begin(), bound_plus.end(), DualComplex{});
            std::fill(bound_minus.begin(), bound_minus.end(), DualComplex{});
        }

        // ---- dual forward, recording the state entering each event ----
        for (std::int64_t event = 0; event < event_count; ++event) {
            DualComplex* slot = trajectory.data()
                + static_cast<std::size_t>(event) * stride;
            std::copy(fplus.begin(), fplus.end(), slot);
            std::copy(fminus.begin(), fminus.end(), slot + states);
            std::copy(longitudinal.begin(), longitudinal.end(), slot + 2U * states);
            if constexpr (TWO_POOL || THREE) {
                std::copy(bound.begin(), bound.end(), slot + 3U * states);
            }
            if constexpr (PAIRED) {
                std::copy(bound_plus.begin(), bound_plus.end(), slot + 4U * states);
                std::copy(
                    bound_minus.begin(), bound_minus.end(), slot + 5U * states
                );
            }
            if constexpr (THREE) {
                std::copy(semisolid.begin(), semisolid.end(), slot + 6U * states);
            }

            const DualFloat dt{
                view.duration[event], dot_duration[event]
            };
            damping.set(damping_rate, dt);
            const DualFloat wout = washout_out(washout_rate, dt);
            const DualFloat dry1 = dual_exp(DualFloat{0.0F, 0.0F} - (r1 * dt));
            const DualFloat dry2 = dual_exp(DualFloat{0.0F, 0.0F} - (r2 * dt));
            const DualFloat e1 = dry1 * wout;
            const DualFloat e2 = dry2 * wout;
            const ThreePoolStep<DualDouble> triple = THREE
                ? three_pool_step<DualDouble>(
                    widen_dual(r1), widen_dual(r1_bound),
                    widen_dual(r1_semisolid), widen_dual(exchange),
                    widen_dual(semisolid_exchange), widen_dual(bound_fraction),
                    widen_dual(semisolid_fraction), widen_dual(dt),
                    widen_dual(wout)
                )
                : ThreePoolStep<DualDouble>{};
            const TwoPoolStep<DualFloat> pools = TWO_POOL
                ? two_pool_step(r1, r1_bound, exchange, bound_fraction, dt, wout)
                : TwoPoolStep<DualFloat>{};
            const TwoPoolTransverse<DualComplex> across = PAIRED
                ? two_pool_transverse_step<DualFloat, DualComplex>(
                    r2, r2_bound, exchange, bound_fraction, transverse_free, pool_shift, dt, wout
                )
                : TwoPoolTransverse<DualComplex>{};
            const DualFloat angle = -2.0F * PI * (b0 * dt);
            const DualComplex phase = dual_polar(angle);
            const DualComplex off = e2 * phase;
            const DualComplex off_conj = conjugate(off);
            for (std::size_t state = 0; state < states; ++state) {
                const DualFloat damp_transverse = damping.transverse[state];
                DualFloat turn_longitudinal{};
                DualFloat turn_transverse{};
                flow_turn_dual(
                    flow_rate, dt, state, turn_longitudinal, turn_transverse
                );
                const DualComplex spin_transverse = dual_polar(turn_transverse);
                const DualComplex spin_longitudinal = dual_polar(turn_longitudinal);
                if constexpr (PAIRED) {
                    const DualComplex carried =
                        phase * damp_transverse * spin_transverse;
                    const DualComplex free_plus = fplus[state];
                    const DualComplex pool_plus = bound_plus[state];
                    const DualComplex free_minus = fminus[state];
                    const DualComplex pool_minus = bound_minus[state];
                    fplus[state] =
                        (across.e11 * free_plus + across.e12 * pool_plus) * carried;
                    bound_plus[state] =
                        (across.e21 * free_plus + across.e22 * pool_plus) * carried;
                    const DualComplex conjugated = conjugate(carried);
                    fminus[state] = (conjugate(across.e11) * free_minus
                        + conjugate(across.e12) * pool_minus) * conjugated;
                    bound_minus[state] = (conjugate(across.e21) * free_minus
                        + conjugate(across.e22) * pool_minus) * conjugated;
                } else {
                    fplus[state] =
                        off * fplus[state] * damp_transverse * spin_transverse;
                    fminus[state] = off_conj * fminus[state] * damp_transverse
                        * conjugate(spin_transverse);
                }
                const DualComplex spin =
                    damping.longitudinal[state] * spin_longitudinal;
                if constexpr (THREE) {
                    const DualComplex pools_in[3] = {
                        longitudinal[state], bound[state], semisolid[state]
                    };
                    DualComplex mixed[3];
                    for (int row = 0; row < 3; ++row) {
                        DualComplex carried{};
                        for (int column = 0; column < 3; ++column) {
                            carried = carried
                                + narrow_dual(triple.entry[row][column])
                                    * pools_in[column];
                        }
                        mixed[row] = carried * spin;
                    }
                    longitudinal[state] = mixed[0];
                    bound[state] = mixed[1];
                    semisolid[state] = mixed[2];
                } else if constexpr (TWO_POOL) {
                    const DualComplex free_state = longitudinal[state];
                    const DualComplex bound_state = bound[state];
                    longitudinal[state] =
                        (pools.e11 * free_state + pools.e12 * bound_state) * spin;
                    bound[state] =
                        (pools.e21 * free_state + pools.e22 * bound_state) * spin;
                } else {
                    longitudinal[state] = e1 * longitudinal[state] * spin;
                }
            }
            if constexpr (THREE) {
                DualComplex* const seats[3] = {
                    &longitudinal[0], &bound[0], &semisolid[0]
                };
                for (int row = 0; row < 3; ++row) {
                    const DualFloat grown = narrow_dual(triple.recovery[row]);
                    seats[row]->value += Complex(grown.value, 0.0F);
                    seats[row]->tangent += Complex(grown.tangent, 0.0F);
                }
            } else if constexpr (TWO_POOL || THREE) {
                longitudinal[0].value += Complex(pools.recovery_free.value, 0.0F);
                longitudinal[0].tangent +=
                    Complex(pools.recovery_free.tangent, 0.0F);
                bound[0].value += Complex(pools.recovery_bound.value, 0.0F);
                bound[0].tangent += Complex(pools.recovery_bound.tangent, 0.0F);
            } else {
                longitudinal[0].value += Complex(1.0F - e1.value, 0.0F);
                longitudinal[0].tangent -= Complex(e1.tangent, 0.0F);
            }

            const std::uint8_t action = primal.action[event];
            if ((action & PRE_SHIFT) != 0) {
                shift(fplus, fminus);
                if constexpr (PAIRED) {
                    shift(bound_plus, bound_minus);
                }
            }
            if (primal.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    const DualFloat negated = DualFloat{0.0F, 0.0F} - efficiency;
                    for (DualComplex& value : longitudinal) {
                        value = negated * value;
                    }
                    if constexpr (PAIRED) {
                        // The exchanging pool is free water and turns over; the
                        // semisolid pool is saturated instead, which is a
                        // different block on a different vector.
                        for (DualComplex& value : bound) {
                            value = negated * value;
                        }
                    }
                } else {
                    const std::int64_t transmit =
                        shimmed ? transmit_row(primal, event, atom) : atom;
                    const DualFloat alpha =
                        DualFloat{view.flip[event], dot_flip[event]}
                        * shim_dual(
                            shimmed, primal.b1, buffers.dot_b1, transmit, b1
                        );
                    const DualFloat phi =
                        DualFloat{view.phase[event], dot_phase[event]}
                        + shim_dual(
                            shimmed, primal.b1_phase, buffers.dot_b1_phase,
                            transmit, b1_phase
                        );
                    if constexpr (SATURATED) {
                        const DualFloat offset =
                            DualFloat{primal.rf_frequency[event], 0.0F} - b0;
                        const DualFloat absorbed = dual_exp(
                            primal.saturation[event]
                            * (alpha * alpha * lineshape_at(primal, offset))
                        );
                        for (DualComplex& value : (THREE ? semisolid : bound)) {
                            value = absorbed * value;
                        }
                    }
                    if constexpr (MODE != RfMode::INSTANT) {
                        DualComplex pair_a{};
                        DualComplex pair_b{};
                        DualComplex slope_a{};
                        DualComplex slope_b{};
                        if constexpr (MODE == RfMode::DYNAMIC) {
                            dynamic_pair_dual_at(
                                primal, buffers.dynamic,
                                dynamic_row(primal, view, event),
                                atom, pair_a, pair_b
                            );
                        } else {
                            profile_pair_slope_dual(
                                primal, table_row<MODE>(primal, event, location),
                                alpha, pair_a, pair_b, slope_a, slope_b
                            );
                        }
                        const DualComplex spun =
                            pair_b * dual_polar(DualFloat{0.0F, 0.0F} - phi);
                        rotate_spinor(
                            fplus, fminus, longitudinal, pair_a, spun
                        );
                        if constexpr (PAIRED) {
                            rotate_spinor(
                                bound_plus, bound_minus, bound, pair_a, spun
                            );
                        }
                    } else {
                        rotate_dual(fplus, fminus, longitudinal, alpha, phi);
                        if constexpr (PAIRED) {
                            rotate_dual(
                                bound_plus, bound_minus, bound, alpha, phi
                            );
                        }
                    }
                }
            }
            if ((action & POST_SHIFT) != 0) {
                shift(fplus, fminus);
                if constexpr (PAIRED) {
                    shift(bound_plus, bound_minus);
                }
            }
            if ((action & SPOIL_AFTER) != 0) {
                std::fill(fplus.begin(), fplus.end(), DualComplex{});
                std::fill(fminus.begin(), fminus.end(), DualComplex{});
                if constexpr (PAIRED) {
                    std::fill(
                        bound_plus.begin(), bound_plus.end(), DualComplex{}
                    );
                    std::fill(
                        bound_minus.begin(), bound_minus.end(), DualComplex{}
                    );
                }
            } else if ((action & SHIFT_AFTER) != 0) {
                shift(fplus, fminus);
                if constexpr (PAIRED) {
                    shift(bound_plus, bound_minus);
                }
            }
        }

        // ---- dual reverse ----
        std::fill(fplus_bar.begin(), fplus_bar.end(), DualComplex{});
        std::fill(fminus_bar.begin(), fminus_bar.end(), DualComplex{});
        std::fill(longitudinal_bar.begin(), longitudinal_bar.end(), DualComplex{});
        if constexpr (TWO_POOL || THREE) {
            std::fill(bound_bar.begin(), bound_bar.end(), DualComplex{});
        }
        if constexpr (PAIRED) {
            std::fill(bound_plus_bar.begin(), bound_plus_bar.end(), DualComplex{});
            std::fill(
                bound_minus_bar.begin(), bound_minus_bar.end(), DualComplex{}
            );
        }
        if constexpr (THREE) {
            std::fill(semisolid_bar.begin(), semisolid_bar.end(), DualComplex{});
        }
        DualFloat grad_t2_bound{0.0F, 0.0F};
        DualFloat grad_pool_shift{0.0F, 0.0F};
        DualFloat grad_t1_bound{0.0F, 0.0F};
        DualFloat grad_t1_semisolid{0.0F, 0.0F};
        DualFloat grad_semisolid_exchange{0.0F, 0.0F};
        DualFloat grad_semisolid_fraction{0.0F, 0.0F};
        DualFloat grad_exchange{0.0F, 0.0F};
        DualFloat grad_bound_fraction{0.0F, 0.0F};
        DualFloat grad_t1{0.0F, 0.0F};
        DualFloat grad_t2{0.0F, 0.0F};
        DualFloat grad_m0{0.0F, 0.0F};
        DualFloat grad_b1{0.0F, 0.0F};
        DualFloat grad_b1_phase{0.0F, 0.0F};
        // Flushed to its shim's row whenever the walk back reaches a pulse on
        // a different one; see the first-order kernel.
        std::int64_t held = 0;
        DualFloat grad_b0{0.0F, 0.0F};
        DualFloat grad_efficiency{0.0F, 0.0F};
        DualFloat grad_damping{0.0F, 0.0F};
        DualFloat grad_flow{0.0F, 0.0F};
        DualFloat grad_washout{0.0F, 0.0F};

        for (std::int64_t event = event_count - 1; event >= 0; --event) {
            const DualComplex* slot = trajectory.data()
                + static_cast<std::size_t>(event) * stride;
            std::copy(slot, slot + states, fplus.begin());
            std::copy(slot + states, slot + 2U * states, fminus.begin());
            std::copy(slot + 2U * states, slot + 3U * states, longitudinal.begin());
            if constexpr (TWO_POOL || THREE) {
                std::copy(slot + 3U * states, slot + 4U * states, bound.begin());
            }
            if constexpr (PAIRED) {
                std::copy(
                    slot + 4U * states, slot + 5U * states, bound_plus.begin()
                );
                std::copy(
                    slot + 5U * states, slot + 6U * states, bound_minus.begin()
                );
            }
            if constexpr (THREE) {
                std::copy(
                    slot + 6U * states, slot + 7U * states, semisolid.begin()
                );
            }

            const DualFloat dt{
                view.duration[event], dot_duration[event]
            };
            damping.set(damping_rate, dt);
            const DualFloat wout = washout_out(washout_rate, dt);
            const DualFloat dry1 = dual_exp(DualFloat{0.0F, 0.0F} - (r1 * dt));
            const DualFloat dry2 = dual_exp(DualFloat{0.0F, 0.0F} - (r2 * dt));
            const DualFloat e1 = dry1 * wout;
            const DualFloat e2 = dry2 * wout;
            const ThreePoolStep<DualDouble> triple = THREE
                ? three_pool_step<DualDouble>(
                    widen_dual(r1), widen_dual(r1_bound),
                    widen_dual(r1_semisolid), widen_dual(exchange),
                    widen_dual(semisolid_exchange), widen_dual(bound_fraction),
                    widen_dual(semisolid_fraction), widen_dual(dt),
                    widen_dual(wout)
                )
                : ThreePoolStep<DualDouble>{};
            const TwoPoolStep<DualFloat> pools = TWO_POOL
                ? two_pool_step(r1, r1_bound, exchange, bound_fraction, dt, wout)
                : TwoPoolStep<DualFloat>{};
            const TwoPoolTransverse<DualComplex> across = PAIRED
                ? two_pool_transverse_step<DualFloat, DualComplex>(
                    r2, r2_bound, exchange, bound_fraction, transverse_free, pool_shift, dt, wout
                )
                : TwoPoolTransverse<DualComplex>{};
            const DualFloat angle = -2.0F * PI * (b0 * dt);
            const DualComplex phase = dual_polar(angle);
            const DualComplex off = e2 * phase;
            const DualComplex off_conj = conjugate(off);
            const std::uint8_t action = primal.action[event];

            for (std::size_t state = 0; state < states; ++state) {
                const DualFloat damp_transverse = damping.transverse[state];
                DualFloat turn_longitudinal{};
                DualFloat turn_transverse{};
                flow_turn_dual(
                    flow_rate, dt, state, turn_longitudinal, turn_transverse
                );
                const DualComplex spin_transverse = dual_polar(turn_transverse);
                const DualComplex spin_longitudinal = dual_polar(turn_longitudinal);
                if constexpr (PAIRED) {
                    const DualComplex carried =
                        phase * damp_transverse * spin_transverse;
                    const DualComplex free_plus = fplus[state];
                    const DualComplex pool_plus = bound_plus[state];
                    const DualComplex free_minus = fminus[state];
                    const DualComplex pool_minus = bound_minus[state];
                    fplus_relaxed[state] =
                        (across.e11 * free_plus + across.e12 * pool_plus) * carried;
                    bound_plus_relaxed[state] =
                        (across.e21 * free_plus + across.e22 * pool_plus) * carried;
                    const DualComplex conjugated = conjugate(carried);
                    fminus_relaxed[state] = (conjugate(across.e11) * free_minus
                        + conjugate(across.e12) * pool_minus) * conjugated;
                    bound_minus_relaxed[state] = (conjugate(across.e21) * free_minus
                        + conjugate(across.e22) * pool_minus) * conjugated;
                } else {
                    fplus_relaxed[state] =
                        off * fplus[state] * damp_transverse * spin_transverse;
                    fminus_relaxed[state] = off_conj * fminus[state]
                        * damp_transverse * conjugate(spin_transverse);
                }
                const DualComplex spin =
                    damping.longitudinal[state] * spin_longitudinal;
                if constexpr (THREE) {
                    const DualComplex pools_in[3] = {
                        longitudinal[state], bound[state], semisolid[state]
                    };
                    DualComplex mixed[3];
                    for (int row = 0; row < 3; ++row) {
                        DualComplex carried{};
                        for (int column = 0; column < 3; ++column) {
                            carried = carried
                                + narrow_dual(triple.entry[row][column])
                                    * pools_in[column];
                        }
                        mixed[row] = carried * spin;
                    }
                    longitudinal_relaxed[state] = mixed[0];
                    bound_relaxed[state] = mixed[1];
                    semisolid_relaxed[state] = mixed[2];
                } else if constexpr (TWO_POOL) {
                    const DualComplex free_state = longitudinal[state];
                    const DualComplex bound_state = bound[state];
                    longitudinal_relaxed[state] =
                        (pools.e11 * free_state + pools.e12 * bound_state) * spin;
                    bound_relaxed[state] =
                        (pools.e21 * free_state + pools.e22 * bound_state) * spin;
                } else {
                    longitudinal_relaxed[state] = e1 * longitudinal[state] * spin;
                }
            }
            if constexpr (THREE) {
                DualComplex* const seats[3] = {
                    &longitudinal_relaxed[0], &bound_relaxed[0], &semisolid_relaxed[0]
                };
                for (int row = 0; row < 3; ++row) {
                    const DualFloat grown = narrow_dual(triple.recovery[row]);
                    seats[row]->value += Complex(grown.value, 0.0F);
                    seats[row]->tangent += Complex(grown.tangent, 0.0F);
                }
            } else if constexpr (TWO_POOL || THREE) {
                longitudinal_relaxed[0].value +=
                    Complex(pools.recovery_free.value, 0.0F);
                longitudinal_relaxed[0].tangent +=
                    Complex(pools.recovery_free.tangent, 0.0F);
                bound_relaxed[0].value += Complex(pools.recovery_bound.value, 0.0F);
                bound_relaxed[0].tangent +=
                    Complex(pools.recovery_bound.tangent, 0.0F);
            } else {
                longitudinal_relaxed[0].value += Complex(1.0F - e1.value, 0.0F);
                longitudinal_relaxed[0].tangent -= Complex(e1.tangent, 0.0F);
            }

            fplus_shifted = fplus_relaxed;
            fminus_shifted = fminus_relaxed;
            if constexpr (PAIRED) {
                bound_plus_shifted = bound_plus_relaxed;
                bound_minus_shifted = bound_minus_relaxed;
            }
            if ((action & PRE_SHIFT) != 0) {
                shift(fplus_shifted, fminus_shifted);
                if constexpr (PAIRED) {
                    shift(bound_plus_shifted, bound_minus_shifted);
                }
            }

            if ((action & SPOIL_AFTER) != 0) {
                std::fill(fplus_bar.begin(), fplus_bar.end(), DualComplex{});
                std::fill(fminus_bar.begin(), fminus_bar.end(), DualComplex{});
                if constexpr (PAIRED) {
                    std::fill(
                        bound_plus_bar.begin(), bound_plus_bar.end(), DualComplex{}
                    );
                    std::fill(
                        bound_minus_bar.begin(), bound_minus_bar.end(),
                        DualComplex{}
                    );
                }
            } else if ((action & SHIFT_AFTER) != 0) {
                shift_adjoint(fplus_bar, fminus_bar);
                if constexpr (PAIRED) {
                    shift_adjoint(bound_plus_bar, bound_minus_bar);
                }
            }
            if ((action & POST_SHIFT) != 0) {
                shift_adjoint(fplus_bar, fminus_bar);
                if constexpr (PAIRED) {
                    shift_adjoint(bound_plus_bar, bound_minus_bar);
                }
            }

            if (primal.kind[event] == 2 && (action & RECORD) != 0) {
                const std::int64_t output = primal.output_index[event];
                const std::int64_t index = view.output_base + output;
                const DualComplex seed{
                    Complex(
                        buffers.grad_output_real[index],
                        buffers.grad_output_imag[index]
                    ),
                    Complex{},
                };
                const DualFloat adc_phase{
                    view.phase[event], dot_phase[event]
                };
                const DualComplex demodulation =
                    dual_polar(DualFloat{0.0F, 0.0F} - adc_phase);
                const DualComplex recorded = PAIRED
                    ? fplus_shifted[0] + bound_plus_shifted[0]
                    : fplus_shifted[0];
                grad_m0 = grad_m0
                    + real_part(conjugate(seed) * recorded * demodulation);
                grad_phase_train[event] = grad_phase_train[event]
                    + real_part(
                        conjugate(seed) * m0 * recorded
                        * DualComplex{Complex(0.0F, -1.0F), Complex{}}
                        * demodulation
                    );
                const DualComplex weighted = conjugate(m0 * demodulation) * seed;
                fplus_bar[0] = fplus_bar[0] + weighted;
                if constexpr (PAIRED) {
                    bound_plus_bar[0] = bound_plus_bar[0] + weighted;
                }
            }

            if (primal.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    const DualFloat negated = DualFloat{0.0F, 0.0F} - efficiency;
                    for (std::size_t state = 0; state < states; ++state) {
                        grad_efficiency = grad_efficiency
                            + real_part(
                                conjugate(longitudinal_bar[state])
                                * (DualComplex{Complex(-1.0F, 0.0F), Complex{}}
                                   * longitudinal_relaxed[state])
                            );
                        longitudinal_bar[state] = negated * longitudinal_bar[state];
                        if constexpr (PAIRED) {
                            grad_efficiency = grad_efficiency
                                + real_part(
                                    conjugate(bound_bar[state])
                                    * (DualComplex{
                                           Complex(-1.0F, 0.0F), Complex{}
                                       }
                                       * bound_relaxed[state])
                                );
                            bound_bar[state] = negated * bound_bar[state];
                        }
                    }
                } else {
                    const DualFloat flip_value{
                        view.flip[event], dot_flip[event]
                    };
                    const std::int64_t transmit =
                        shimmed ? transmit_row(primal, event, atom) : atom;
                    const DualFloat pulse_b1 = shim_dual(
                        shimmed, primal.b1, buffers.dot_b1, transmit, b1
                    );
                    const DualFloat alpha = flip_value * pulse_b1;
                    const DualFloat phi =
                        DualFloat{view.phase[event], dot_phase[event]}
                        + shim_dual(
                            shimmed, primal.b1_phase, buffers.dot_b1_phase,
                            transmit, b1_phase
                        );
                    const std::int64_t row =
                        shimmed ? primal.shim_index[event] : 0;
                    if (row != held) {
                        DualFloat& magnitude = grad_tissue_local[
                            (layout.base[B1_INDEX] + held) * atoms + atom
                        ];
                        DualFloat& angle = grad_tissue_local[
                            (layout.base[B1_PHASE_INDEX] + held) * atoms + atom
                        ];
                        magnitude = magnitude + grad_b1;
                        angle = angle + grad_b1_phase;
                        grad_b1 = DualFloat{0.0F, 0.0F};
                        grad_b1_phase = DualFloat{0.0F, 0.0F};
                        held = row;
                    }
                    DualFloat grad_alpha{0.0F, 0.0F};
                    DualFloat grad_phi{0.0F, 0.0F};
                    if constexpr (MODE != RfMode::INSTANT) {
                        DualComplex pair_a{};
                        DualComplex pair_b{};
                        DualComplex slope_a{};
                        DualComplex slope_b{};
                        if constexpr (MODE == RfMode::DYNAMIC) {
                            dynamic_pair_dual_at(
                                primal, buffers.dynamic,
                                dynamic_row(primal, view, event),
                                atom, pair_a, pair_b
                            );
                        } else {
                            profile_pair_slope_dual(
                                primal, table_row<MODE>(primal, event, location),
                                alpha, pair_a, pair_b, slope_a, slope_b
                            );
                        }
                        const DualComplex turn =
                            dual_polar(DualFloat{0.0F, 0.0F} - phi);
                        const DualComplex spun = pair_b * turn;
                        DualComplex grad_a{};
                        DualComplex grad_b{};
                        rotate_adjoint_spinor_dual(
                            fplus_shifted,
                            fminus_shifted,
                            longitudinal_relaxed,
                            fplus_bar,
                            fminus_bar,
                            longitudinal_bar,
                            pair_a,
                            spun,
                            grad_a,
                            grad_b
                        );
                        if constexpr (PAIRED) {
                            rotate_adjoint_spinor_dual(
                                bound_plus_shifted,
                                bound_minus_shifted,
                                bound_relaxed,
                                bound_plus_bar,
                                bound_minus_bar,
                                bound_bar,
                                pair_a,
                                spun,
                                grad_a,
                                grad_b
                            );
                        }
                        grad_phi = real_part(
                            conjugate(grad_b) * (Complex(0.0F, -1.0F) * spun)
                        );
                        if constexpr (MODE == RfMode::DYNAMIC) {
                            // The flip is inside the pair, so the cotangent
                            // goes out on the rotation -- value and direction
                            // together, since this pass carries both.
                            const std::int64_t slot = dynamic_offset(
                                primal,
                                dynamic_row(primal, view, event),
                                atom
                            );
                            float* const held_value =
                                buffers.grad_dot_dynamic + slot;
                            float* const held_tangent =
                                buffers.grad_dynamic + slot;
                            // ``b`` was turned by the phase after the pair came
                            // out, so the cotangent turns back the other way.
                            const DualComplex held = grad_b * conjugate(turn);
                            const DualFloat parts[4] = {
                                real_part(grad_a), imag_part(grad_a),
                                real_part(held), imag_part(held),
                            };
                            for (std::size_t part = 0; part < 4; ++part) {
                                held_value[part] += parts[part].value;
                                held_tangent[part] += parts[part].tangent;
                            }
                        } else {
                            grad_alpha =
                                real_part(conjugate(grad_a) * slope_a)
                                + real_part(
                                    conjugate(grad_b) * (slope_b * turn)
                                );
                        }
                    } else {
                        rotate_adjoint_dual(
                            fplus_shifted,
                            fminus_shifted,
                            longitudinal_relaxed,
                            fplus_bar,
                            fminus_bar,
                            longitudinal_bar,
                            alpha,
                            phi,
                            grad_alpha,
                            grad_phi
                        );
                        if constexpr (PAIRED) {
                            rotate_adjoint_dual(
                                bound_plus_shifted,
                                bound_minus_shifted,
                                bound_relaxed,
                                bound_plus_bar,
                                bound_minus_bar,
                                bound_bar,
                                alpha,
                                phi,
                                grad_alpha,
                                grad_phi
                            );
                        }
                    }
                    if constexpr (SATURATED) {
                        // The pulse scales every order of the semisolid pool by
                        // one real number, so its cotangent is a single sum over
                        // the states it multiplied.
                        DualState& absorbing_bar =
                            THREE ? semisolid_bar : bound_bar;
                        DualState& absorbing_relaxed =
                            THREE ? semisolid_relaxed : bound_relaxed;
                        const DualFloat offset =
                            DualFloat{primal.rf_frequency[event], 0.0F} - b0;
                        DualFloat shape{};
                        DualFloat shape_slope{};
                        lineshape_at_slope(primal, offset, shape, shape_slope);
                        const DualFloat power =
                            primal.saturation[event] * (alpha * alpha);
                        const DualFloat absorbed = dual_exp(power * shape);
                        DualFloat grad_absorbed{0.0F, 0.0F};
                        for (std::size_t state = 0; state < states; ++state) {
                            grad_absorbed = grad_absorbed
                                + real_part(
                                    conjugate(absorbing_bar[state])
                                    * absorbing_relaxed[state]
                                );
                            absorbing_bar[state] = absorbed * absorbing_bar[state];
                        }
                        const DualFloat grad_exponent = grad_absorbed * absorbed;
                        grad_alpha = grad_alpha
                            + (primal.saturation[event] * 2.0F)
                                * (grad_exponent * (alpha * shape));
                        grad_b0 = grad_b0
                            - grad_exponent * (power * shape_slope);
                    }
                    grad_flip_train[event] =
                        grad_flip_train[event] + grad_alpha * pulse_b1;
                    grad_b1 = grad_b1 + grad_alpha * flip_value;
                    grad_phase_train[event] = grad_phase_train[event] + grad_phi;
                    grad_b1_phase = grad_b1_phase + grad_phi;
                }
            }

            if ((action & PRE_SHIFT) != 0) {
                shift_adjoint(fplus_bar, fminus_bar);
                if constexpr (PAIRED) {
                    shift_adjoint(bound_plus_bar, bound_minus_bar);
                }
            }

            DualFloat grad_e1{0.0F, 0.0F};
            DualFloat grad_e2{0.0F, 0.0F};
            DualFloat grad_angle{0.0F, 0.0F};
            DualFloat grad_b_factor{0.0F, 0.0F};
            DualFloat grad_turn{0.0F, 0.0F};
            DualFloat grad_e11{0.0F, 0.0F};
            DualFloat grad_e12{0.0F, 0.0F};
            DualFloat grad_e21{0.0F, 0.0F};
            DualFloat grad_e22{0.0F, 0.0F};
            DualComplex grad_across_11{};
            DualComplex grad_across_12{};
            DualComplex grad_across_21{};
            DualComplex grad_across_22{};
            const Complex imaginary(0.0F, 1.0F);
            const DualFloat grad_recovery_free =
                (TWO_POOL || THREE) ? real_part(longitudinal_bar[0]) : DualFloat{};
            const DualFloat grad_recovery_bound =
                (TWO_POOL || THREE) ? real_part(bound_bar[0]) : DualFloat{};
            const DualFloat grad_recovery_semisolid =
                THREE ? real_part(semisolid_bar[0]) : DualFloat{};
            // The nine entries of the three-pool operator, summed over the
            // orders that share them and pushed back through the closed form
            // once for the whole interval.
            DualFloat grad_triple[3][3]{};
            if constexpr (!TWO_POOL && !THREE) {
                grad_e1 = grad_e1 - real_part(longitudinal_bar[0]);
            }
            for (std::size_t state = 0; state < states; ++state) {
                const DualComplex ap = fplus_bar[state];
                const DualComplex am = fminus_bar[state];
                const DualComplex az = longitudinal_bar[state];
                const DualFloat damp_transverse = damping.transverse[state];
                const DualFloat damp_longitudinal = damping.longitudinal[state];
                DualFloat turn_longitudinal{};
                DualFloat turn_transverse{};
                flow_turn_dual(
                    flow_rate, dt, state, turn_longitudinal, turn_transverse
                );
                const DualComplex spin_transverse = dual_polar(turn_transverse);
                const DualComplex spin_longitudinal = dual_polar(turn_longitudinal);
                const DualComplex carried =
                    phase * damp_transverse * spin_transverse;
                const DualComplex full_transverse =
                    off * damp_transverse * spin_transverse;
                // The damping is homogeneous of degree one in every transverse
                // state it acts on, so its gradient times the damping itself is
                // the cotangent taken against the states the interval leaves.
                DualFloat transverse_scaled{0.0F, 0.0F};
                DualFloat angle_term{0.0F, 0.0F};
                if constexpr (PAIRED) {
                    const DualComplex abp = bound_plus_bar[state];
                    const DualComplex abm = bound_minus_bar[state];
                    const DualComplex fp = fplus[state];
                    const DualComplex bp = bound_plus[state];
                    const DualComplex fm = fminus[state];
                    const DualComplex bm = bound_minus[state];
                    const DualComplex out_fp = fplus_relaxed[state];
                    const DualComplex out_bp = bound_plus_relaxed[state];
                    const DualComplex out_fm = fminus_relaxed[state];
                    const DualComplex out_bm = bound_minus_relaxed[state];
                    const DualComplex plus_side =
                        conjugate(ap) * out_fp + conjugate(abp) * out_bp;
                    const DualComplex minus_side =
                        conjugate(am) * out_fm + conjugate(abm) * out_bm;
                    transverse_scaled = real_part(plus_side + minus_side);
                    angle_term =
                        real_part(imaginary * (plus_side - minus_side));
                    // ``F-`` follows the conjugate of the operator, so its
                    // cotangent lands on the entry itself rather than on the
                    // conjugate of it.
                    grad_across_11 = grad_across_11
                        + (conjugate(ap) * fp + am * conjugate(fm)) * carried;
                    grad_across_12 = grad_across_12
                        + (conjugate(ap) * bp + am * conjugate(bm)) * carried;
                    grad_across_21 = grad_across_21
                        + (conjugate(abp) * fp + abm * conjugate(fm)) * carried;
                    grad_across_22 = grad_across_22
                        + (conjugate(abp) * bp + abm * conjugate(bm)) * carried;
                    const DualComplex step_11 = across.e11 * carried;
                    const DualComplex step_12 = across.e12 * carried;
                    const DualComplex step_21 = across.e21 * carried;
                    const DualComplex step_22 = across.e22 * carried;
                    fplus_bar[state] =
                        conjugate(step_11) * ap + conjugate(step_21) * abp;
                    bound_plus_bar[state] =
                        conjugate(step_12) * ap + conjugate(step_22) * abp;
                    fminus_bar[state] = step_11 * am + step_21 * abm;
                    bound_minus_bar[state] = step_12 * am + step_22 * abm;
                } else {
                    const DualFloat transverse_term = real_part(
                        conjugate(ap) * carried * fplus[state]
                        + conjugate(am) * conjugate(carried) * fminus[state]
                    );
                    grad_e2 = grad_e2 + transverse_term;
                    transverse_scaled = e2 * transverse_term;
                    angle_term =
                        real_part(conjugate(ap) * (imaginary * (full_transverse * fplus[state])))
                        - real_part(
                            conjugate(am)
                            * (imaginary * (conjugate(full_transverse) * fminus[state]))
                        );
                    fplus_bar[state] = conjugate(full_transverse) * ap;
                    fminus_bar[state] = full_transverse * am;
                }
                // A turn of the transverse states and the off-resonance angle
                // are the same derivative; only the weight they carry differs.
                grad_angle = grad_angle + angle_term;
                const DualComplex spin = damp_longitudinal * spin_longitudinal;
                DualFloat longitudinal_damp_term{0.0F, 0.0F};
                DualFloat longitudinal_angle_term{0.0F, 0.0F};
                if constexpr (THREE) {
                    const DualComplex bars[3] = {
                        az, bound_bar[state], semisolid_bar[state]
                    };
                    const DualComplex pools_in[3] = {
                        longitudinal[state], bound[state], semisolid[state]
                    };
                    DualComplex mixed[3];
                    for (int row = 0; row < 3; ++row) {
                        DualComplex carried{};
                        for (int column = 0; column < 3; ++column) {
                            carried = carried
                                + narrow_dual(triple.entry[row][column])
                                    * pools_in[column];
                            grad_triple[row][column] = grad_triple[row][column]
                                + real_part(
                                    conjugate(bars[row])
                                    * (spin * pools_in[column])
                                );
                        }
                        mixed[row] = carried;
                    }
                    for (int row = 0; row < 3; ++row) {
                        longitudinal_damp_term = longitudinal_damp_term
                            + real_part(conjugate(bars[row]) * (spin * mixed[row]));
                        longitudinal_angle_term = longitudinal_angle_term
                            + real_part(
                                conjugate(bars[row])
                                * (imaginary * (spin * mixed[row]))
                            );
                    }
                    DualComplex back[3];
                    for (int column = 0; column < 3; ++column) {
                        DualComplex carried{};
                        for (int row = 0; row < 3; ++row) {
                            carried = carried
                                + conjugate(
                                    narrow_dual(triple.entry[row][column]) * spin
                                ) * bars[row];
                        }
                        back[column] = carried;
                    }
                    longitudinal_bar[state] = back[0];
                    bound_bar[state] = back[1];
                    semisolid_bar[state] = back[2];
                } else if constexpr (TWO_POOL) {
                    const DualComplex ab = bound_bar[state];
                    const DualComplex free_state = longitudinal[state];
                    const DualComplex bound_state = bound[state];
                    const DualComplex mixed_free =
                        pools.e11 * free_state + pools.e12 * bound_state;
                    const DualComplex mixed_bound =
                        pools.e21 * free_state + pools.e22 * bound_state;
                    grad_e11 = grad_e11
                        + real_part(conjugate(az) * (spin * free_state));
                    grad_e12 = grad_e12
                        + real_part(conjugate(az) * (spin * bound_state));
                    grad_e21 = grad_e21
                        + real_part(conjugate(ab) * (spin * free_state));
                    grad_e22 = grad_e22
                        + real_part(conjugate(ab) * (spin * bound_state));
                    longitudinal_damp_term = real_part(
                        conjugate(az) * (spin * mixed_free)
                        + conjugate(ab) * (spin * mixed_bound)
                    );
                    longitudinal_angle_term = real_part(
                        conjugate(az) * (imaginary * (spin * mixed_free))
                        + conjugate(ab) * (imaginary * (spin * mixed_bound))
                    );
                    longitudinal_bar[state] =
                        conjugate(pools.e11 * spin) * az
                        + conjugate(pools.e21 * spin) * ab;
                    bound_bar[state] = conjugate(pools.e12 * spin) * az
                        + conjugate(pools.e22 * spin) * ab;
                } else {
                    const DualComplex full_longitudinal = e1 * spin;
                    const DualFloat longitudinal_term =
                        real_part(conjugate(az) * (spin * longitudinal[state]));
                    grad_e1 = grad_e1 + longitudinal_term;
                    longitudinal_damp_term = e1 * longitudinal_term;
                    longitudinal_angle_term = real_part(
                        conjugate(az)
                        * (imaginary * (full_longitudinal * longitudinal[state]))
                    );
                    longitudinal_bar[state] = conjugate(full_longitudinal) * az;
                }
                grad_b_factor = grad_b_factor
                    - transverse_weight(state) * transverse_scaled
                    - longitudinal_weight(state) * longitudinal_damp_term;
                grad_turn = grad_turn
                    - (static_cast<float>(state) + 0.5F) * angle_term
                    - static_cast<float>(state) * longitudinal_angle_term;
            }

            DualFloat grad_exchange_attenuation{0.0F, 0.0F};
            DualFloat grad_two_pool_duration{0.0F, 0.0F};
            if constexpr (THREE) {
                DualDouble bar_entry[3][3];
                for (int row = 0; row < 3; ++row) {
                    for (int column = 0; column < 3; ++column) {
                        bar_entry[row][column] = widen_dual(grad_triple[row][column]);
                    }
                }
                const DualDouble bar_recovery[3] = {
                    widen_dual(grad_recovery_free),
                    widen_dual(grad_recovery_bound),
                    widen_dual(grad_recovery_semisolid),
                };
                const ThreePoolGradient<DualDouble> back =
                    three_pool_step_adjoint<DualDouble>(
                        widen_dual(r1), widen_dual(r1_bound),
                        widen_dual(r1_semisolid), widen_dual(exchange),
                        widen_dual(semisolid_exchange), widen_dual(bound_fraction),
                        widen_dual(semisolid_fraction), widen_dual(dt),
                        widen_dual(wout), bar_entry, bar_recovery
                    );
                grad_t1 = grad_t1 - narrow_dual(back.r1_free)
                    * (1000.0F * dual_reciprocal(t1 * t1));
                grad_t1_bound = grad_t1_bound - narrow_dual(back.r1_pool_b)
                    * (1000.0F * dual_reciprocal(t1_bound * t1_bound));
                grad_t1_semisolid = grad_t1_semisolid - narrow_dual(back.r1_bound)
                    * (1000.0F * dual_reciprocal(t1_semisolid * t1_semisolid));
                grad_exchange = grad_exchange + narrow_dual(back.exchange_b);
                grad_semisolid_exchange =
                    grad_semisolid_exchange + narrow_dual(back.exchange_c);
                grad_bound_fraction =
                    grad_bound_fraction + narrow_dual(back.fraction_b);
                grad_semisolid_fraction =
                    grad_semisolid_fraction + narrow_dual(back.fraction_c);
                grad_exchange_attenuation =
                    grad_exchange_attenuation + narrow_dual(back.attenuation);
                grad_two_pool_duration =
                    grad_two_pool_duration + narrow_dual(back.dt);
            }
            if constexpr (TWO_POOL) {
                const TwoPoolGradient<DualFloat> back = two_pool_step_adjoint(
                    r1, r1_bound, exchange, bound_fraction, dt, wout,
                    grad_e11, grad_e12, grad_e21, grad_e22,
                    grad_recovery_free, grad_recovery_bound
                );
                grad_t1 = grad_t1
                    - back.r1_free * (1000.0F * dual_reciprocal(t1 * t1));
                grad_t1_bound = grad_t1_bound
                    - back.r1_bound
                        * (1000.0F * dual_reciprocal(t1_bound * t1_bound));
                grad_exchange = grad_exchange + back.exchange;
                grad_bound_fraction = grad_bound_fraction + back.bound;
                grad_exchange_attenuation = back.attenuation;
                grad_two_pool_duration = back.dt;
            }
            if constexpr (PAIRED) {
                const TwoPoolTransverseGradient<DualFloat> across_back =
                    two_pool_transverse_adjoint<DualFloat, DualComplex>(
                        r2, r2_bound, exchange, bound_fraction, transverse_free,
                        pool_shift, dt, wout, grad_across_11, grad_across_12,
                        grad_across_21, grad_across_22
                    );
                grad_t2 = grad_t2
                    - across_back.r2_free * (1000.0F * dual_reciprocal(t2 * t2));
                grad_t2_bound = grad_t2_bound
                    - across_back.r2_bound
                        * (1000.0F * dual_reciprocal(t2_bound * t2_bound));
                grad_exchange = grad_exchange + across_back.exchange;
                // The free water is what both second pools leave, so a
                // cotangent on it reaches each of their fractions turned over.
                grad_bound_fraction =
                    grad_bound_fraction + across_back.bound - across_back.free;
                if constexpr (THREE) {
                    grad_semisolid_fraction =
                        grad_semisolid_fraction - across_back.free;
                }
                grad_pool_shift = grad_pool_shift + across_back.shift_hz;
                grad_exchange_attenuation =
                    grad_exchange_attenuation + across_back.attenuation;
                grad_two_pool_duration =
                    grad_two_pool_duration + across_back.dt;
            }

            const DualFloat inverse_t1_squared =
                dual_reciprocal(t1 * t1);
            const DualFloat inverse_t2_squared =
                dual_reciprocal(t2 * t2);
            grad_t1 = grad_t1 + grad_e1 * e1 * (1000.0F * (dt * inverse_t1_squared));
            grad_t2 = grad_t2 + grad_e2 * e2 * (1000.0F * (dt * inverse_t2_squared));
            grad_b0 = grad_b0 + grad_angle * (-2.0F * PI * dt);
            const DualFloat grad_wout =
                washout_rate.value * dt.value < 1.0F
                    ? DualFloat{0.0F, 0.0F}
                        - (dry1 * grad_e1 + dry2 * grad_e2
                           + grad_exchange_attenuation)
                    : DualFloat{0.0F, 0.0F};
            grad_damping = grad_damping + grad_b_factor * dt;
            grad_flow = grad_flow + grad_turn * dt;
            grad_washout = grad_washout + grad_wout * dt;
            grad_duration_train[event] = grad_duration_train[event]
                + (DualFloat{0.0F, 0.0F} - (grad_e1 * (r1 * e1)))
                - (grad_e2 * (r2 * e2))
                + grad_two_pool_duration
                + grad_angle * (-2.0F * PI * b0)
                + grad_b_factor * damping_rate
                + grad_turn * flow_rate
                + grad_wout * washout_rate;
        }

        if constexpr (TWO_POOL || THREE) {
            grad_bound_fraction = grad_bound_fraction
                + real_part(bound_bar[0]) - real_part(longitudinal_bar[0]);
        }
        if constexpr (THREE) {
            grad_semisolid_fraction = grad_semisolid_fraction
                + real_part(semisolid_bar[0]) - real_part(longitudinal_bar[0]);
        }

        // A run carries one second pool or none, so the rows of the other are
        // left at zero -- the true gradient, the kernel not having read them.
        const DualFloat contributions[TISSUE_COUNT] = {
            grad_t1, grad_t2, grad_m0, grad_b1, grad_b1_phase, grad_b0,
            grad_efficiency, grad_damping,
            primal.flow_scale * grad_flow
                + (speed_direction(velocity) * primal.washout_scale)
                    * grad_washout,
            MT ? grad_bound_fraction
               : (THREE ? grad_semisolid_fraction : DualFloat{}),
            MT ? grad_exchange
               : (THREE ? grad_semisolid_exchange : DualFloat{}),
            MT ? grad_t1_bound : (THREE ? grad_t1_semisolid : DualFloat{}),
            PAIRED ? grad_bound_fraction : DualFloat{},
            PAIRED ? grad_exchange : DualFloat{},
            PAIRED ? grad_t1_bound : DualFloat{},
            PAIRED ? grad_t2_bound : DualFloat{},
            PAIRED ? grad_pool_shift : DualFloat{},
        };
        for (std::size_t parameter = 0; parameter < TISSUE_COUNT; ++parameter) {
            const std::size_t plane = layout.base[parameter]
                + (parameter == B1_INDEX || parameter == B1_PHASE_INDEX
                       ? static_cast<std::size_t>(held)
                       : 0U);
            DualFloat& slot = grad_tissue_local[plane * atoms + atom];
            slot = slot + contributions[parameter];
        }
    }
}

// ---------------------------------------------------------------------------
// Tabulated pools.
//
// A tissue with more than one chemically exchanging pool arrives as the
// operators its intervals apply (``Buffers::pool_values``), formed outside in
// double precision. Every exchanging pool carries a transverse pair and a
// longitudinal state of its own, and a semisolid pool, where there is one, a
// longitudinal state alone and last. What a voxel does to all of its pools
// alike -- off-resonance, diffusion, flow and washout -- is applied here as it
// is for a single pool; what separates them is inside the operators.
//
// The four passes are two bodies, each written once over a scalar that is
// either a float or a dual number: the forward pass carrying a direction is
// the JVP, and the adjoint carrying one is the forward-over-reverse pass. The
// properties the operators were formed from take their derivatives through
// the table, so these bodies differentiate the table and nothing it came from.
//
// An event reads the row of its own interval length, which it shares with
// every event of that length. The table holds that row's slope and curvature
// along the interval when the durations are differentiated, and a pass
// following a direction in an event's duration moves the operators it reads
// along that slope.
// ---------------------------------------------------------------------------

inline Complex conjugate(const Complex a) {
    return std::conj(a);
}

inline float imag_part(const Complex a) {
    return a.imag();
}

inline DualComplex& operator+=(DualComplex& a, const DualComplex b) {
    a = a + b;
    return a;
}

inline DualFloat& operator+=(DualFloat& a, const DualFloat b) {
    a = a + b;
    return a;
}

inline float tangent_of(const float) {
    return 0.0F;
}

inline float tangent_of(const DualFloat a) {
    return a.tangent;
}

template <typename Real>
struct Carried;

template <>
struct Carried<float> {
    using Cplx = Complex;
};

template <>
struct Carried<DualFloat> {
    using Cplx = DualComplex;
};

template <typename Real>
Real constant(float value);

template <>
inline float constant<float>(const float value) {
    return value;
}

template <>
inline DualFloat constant<DualFloat>(const float value) {
    return DualFloat{value, 0.0F};
}

// A buffer entry, carrying the direction along it when the pass follows one.
template <typename Real>
Real read_at(const float* value, const float* direction, std::int64_t index);

template <>
inline float read_at<float>(
    const float* const value, const float* const, const std::int64_t index
) {
    return value[index];
}

template <>
inline DualFloat read_at<DualFloat>(
    const float* const value, const float* const direction, const std::int64_t index
) {
    return DualFloat{value[index], direction != nullptr ? direction[index] : 0.0F};
}

// A per-voxel property the launch may leave out, at its identity when it does.
template <typename Real>
inline Real voxel_value(
    const float* const value,
    const float* const direction,
    const std::int64_t atom,
    const bool live,
    const float identity
) {
    return live ? read_at<Real>(value, direction, atom) : constant<Real>(identity);
}

inline float speed_of(const float velocity) {
    return std::fabs(velocity);
}

inline DualFloat speed_of(const DualFloat velocity) {
    return DualFloat{
        std::fabs(velocity.value), speed_direction(velocity.value) * velocity.tangent
    };
}

// ``magnitude * exp(i angle)``, or the magnitude alone when nothing turns.
inline Complex turned_by(const float magnitude, const float angle, const bool turning) {
    return turned(magnitude, angle, turning);
}

inline DualComplex turned_by(
    const DualFloat magnitude, const DualFloat angle, const bool turning
) {
    const Complex unit = turned(1.0F, angle.value, turning);
    const Complex spun =
        turning ? Complex(0.0F, magnitude.value * angle.tangent) : Complex{};
    return DualComplex{
        magnitude.value * unit, (Complex(magnitude.tangent, 0.0F) + spun) * unit
    };
}

template <typename Cplx>
Cplx seeded(float real, float imag);

template <>
inline Complex seeded<Complex>(const float real, const float imag) {
    return Complex(real, imag);
}

template <>
inline DualComplex seeded<DualComplex>(const float real, const float imag) {
    return DualComplex{Complex(real, imag), Complex{}};
}

// Adds a cotangent to its value plane, and to its tangent plane when the pass
// carries a direction and the caller wants that plane.
inline void deposit(float* const value, float* const, const float amount) {
    *value += amount;
}

inline void deposit(float* const value, float* const tangent, const DualFloat amount) {
    *value += amount.value;
    if (tangent != nullptr) {
        *tangent += amount.tangent;
    }
}

// The cotangent an event's interval length puts on the row's slope. At the
// row's own length the slope reaches the output only through a direction in
// that length, so only the tangent plane of a pass following one is written.
inline void deposit_along(float* const, const float, const float) {}

inline void deposit_along(
    float* const tangent, const DualFloat amount, const float along
) {
    *tangent += amount.value * along;
}

inline void rotate_by(
    State& fplus, State& fminus, State& longitudinal, const float alpha,
    const float phi
) {
    rotate(fplus, fminus, longitudinal, alpha, phi);
}

inline void rotate_by(
    DualState& fplus, DualState& fminus, DualState& longitudinal,
    const DualFloat alpha, const DualFloat phi
) {
    rotate_dual(fplus, fminus, longitudinal, alpha, phi);
}

inline void rotate_adjoint_by(
    const State& fplus_in, const State& fminus_in, const State& longitudinal_in,
    State& fplus_bar, State& fminus_bar, State& longitudinal_bar,
    const float alpha, const float phi, float& grad_alpha, float& grad_phi
) {
    rotate_adjoint(
        fplus_in, fminus_in, longitudinal_in, fplus_bar, fminus_bar,
        longitudinal_bar, alpha, phi, grad_alpha, grad_phi
    );
}

inline void rotate_adjoint_by(
    const DualState& fplus_in, const DualState& fminus_in,
    const DualState& longitudinal_in, DualState& fplus_bar, DualState& fminus_bar,
    DualState& longitudinal_bar, const DualFloat alpha, const DualFloat phi,
    DualFloat& grad_alpha, DualFloat& grad_phi
) {
    rotate_adjoint_dual(
        fplus_in, fminus_in, longitudinal_in, fplus_bar, fminus_bar,
        longitudinal_bar, alpha, phi, grad_alpha, grad_phi
    );
}

inline void rotate_adjoint_spinor_by(
    const State& fplus_in, const State& fminus_in, const State& longitudinal_in,
    State& fplus_bar, State& fminus_bar, State& longitudinal_bar,
    const Complex a, const Complex b, Complex& grad_a, Complex& grad_b
) {
    rotate_adjoint_spinor(
        fplus_in, fminus_in, longitudinal_in, fplus_bar, fminus_bar,
        longitudinal_bar, a, b, grad_a, grad_b
    );
}

inline void rotate_adjoint_spinor_by(
    const DualState& fplus_in, const DualState& fminus_in,
    const DualState& longitudinal_in, DualState& fplus_bar, DualState& fminus_bar,
    DualState& longitudinal_bar, const DualComplex a, const DualComplex b,
    DualComplex& grad_a, DualComplex& grad_b
) {
    rotate_adjoint_spinor_dual(
        fplus_in, fminus_in, longitudinal_in, fplus_bar, fminus_bar,
        longitudinal_bar, a, b, grad_a, grad_b
    );
}

// What a tabulated pass reads beside the primal buffers, and where an adjoint
// writes what it returns. A pass following no direction leaves the directions
// null, and a pass that is not an adjoint leaves the rest null.
struct PoolPass {
    const float* m0 = nullptr;
    const float* b1 = nullptr;
    const float* b1_phase = nullptr;
    const float* b0 = nullptr;
    const float* inversion_efficiency = nullptr;
    const float* diffusion = nullptr;
    const float* velocity = nullptr;
    const float* duration = nullptr;
    const float* flip = nullptr;
    const float* phase = nullptr;
    const float* dynamic = nullptr;
    const float* table = nullptr;
    const float* grad_output_real = nullptr;
    const float* grad_output_imag = nullptr;
    // The cotangent on the table, one table per (train, atom) in work order so
    // that every entry has one writer, and on the per-voxel rotations. A pass
    // following a direction returns the first-order cotangent in the value
    // plane and its derivative in the tangent plane.
    float* grad_table = nullptr;
    float* grad_table_tangent = nullptr;
    float* grad_dynamic = nullptr;
    float* grad_dynamic_tangent = nullptr;
};

// One row of the table as an event reads it. A pass following a direction
// moves every entry along the table's own direction and along the row's slope
// by the event's direction in its interval length.
template <typename Real>
struct PoolRow;

template <>
struct PoolRow<float> {
    const float* value;

    float real(const std::size_t index) const { return value[index]; }
};

template <>
struct PoolRow<DualFloat> {
    const float* value;
    const float* direction;
    const float* slope;
    float along;

    DualFloat real(const std::size_t index) const {
        return DualFloat{
            value[index],
            (direction != nullptr ? direction[index] : 0.0F)
                + (slope != nullptr ? slope[index] * along : 0.0F),
        };
    }
};

template <typename Real>
PoolRow<Real> pool_row(
    const PoolShape& shape, const float* table, const float* direction,
    std::int64_t index, std::size_t block, float along
);

template <>
inline PoolRow<float> pool_row<float>(
    const PoolShape& shape, const float* const table, const float* const,
    const std::int64_t index, const std::size_t block, const float
) {
    return PoolRow<float>{table + shape.row(index, block)};
}

template <>
inline PoolRow<DualFloat> pool_row<DualFloat>(
    const PoolShape& shape, const float* const table, const float* const direction,
    const std::int64_t index, const std::size_t block, const float along
) {
    const std::size_t at = shape.row(index, block);
    const bool sloped = block + 1 < shape.blocks;
    return PoolRow<DualFloat>{
        table + at,
        direction != nullptr ? direction + at : nullptr,
        sloped ? table + shape.row(index, block + 1) : nullptr,
        along,
    };
}

// One interval's operators, read out of a row once for every order.
template <typename Real>
struct PoolOperators {
    using Cplx = typename Carried<Real>::Cplx;

    std::vector<Real> longitudinal;
    std::vector<Real> restored;
    std::vector<Cplx> transverse;

    explicit PoolOperators(const PoolShape& shape)
        : longitudinal(shape.longitudinal * shape.longitudinal),
          restored(shape.longitudinal),
          transverse(shape.transverse * shape.transverse) {}

    void read(const PoolRow<Real>& row, const PoolShape& shape) {
        for (std::size_t index = 0; index < longitudinal.size(); ++index) {
            longitudinal[index] = row.real(index);
        }
        for (std::size_t index = 0; index < restored.size(); ++index) {
            restored[index] = row.real(shape.restored() + index);
        }
        for (std::size_t index = 0; index < transverse.size(); ++index) {
            const std::size_t at = shape.across() + 2U * index;
            transverse[index] = widen(row.real(at)) + as_imaginary(row.real(at + 1U));
        }
    }
};

// What one voxel brings to every event of a train.
template <typename Real>
struct PoolVoxel {
    Real m0;
    Real b1;
    Real b1_phase;
    Real b0;
    Real efficiency;
    Real damping_rate;
    Real velocity;
    Real flow_rate;
    Real washout_rate;

    PoolVoxel(const Buffers& buffers, const PoolPass& pass, const std::int64_t atom)
        : m0(voxel_value<Real>(buffers.m0, pass.m0, atom, buffers.density, 1.0F)),
          b1(voxel_value<Real>(buffers.b1, pass.b1, atom, buffers.transmit, 1.0F)),
          b1_phase(voxel_value<Real>(
              buffers.b1_phase, pass.b1_phase, atom, buffers.off_axis, 0.0F
          )),
          b0(voxel_value<Real>(buffers.b0, pass.b0, atom, buffers.off_axis, 0.0F)),
          efficiency(voxel_value<Real>(
              buffers.inversion_efficiency, pass.inversion_efficiency, atom,
              buffers.inverting, 1.0F
          )),
          damping_rate(voxel_value<Real>(
              buffers.diffusion, pass.diffusion, atom, buffers.diffusing, 0.0F
          )),
          velocity(voxel_value<Real>(
              buffers.velocity, pass.velocity, atom, buffers.moving, 0.0F
          )),
          flow_rate(buffers.flow_scale * velocity),
          washout_rate(buffers.washout_scale * speed_of(velocity)) {}
};

// The factors an interval applies to order ``state`` of every pool alike: the
// transverse one with the voxel's off-resonance and flow turn, the
// longitudinal one with the flow turn. Washout multiplies both.
template <typename Real, typename Cplx>
inline void interval_factors(
    const Damping<Real>& damping,
    const Real angle,
    const Real flow_rate,
    const Real dt,
    const std::size_t state,
    const bool turning,
    const bool flowing,
    Cplx& transverse,
    Cplx& longitudinal
) {
    Real turn_longitudinal = constant<Real>(0.0F);
    Real turn_transverse = constant<Real>(0.0F);
    if (flowing) {
        flow_turn_dual(flow_rate, dt, state, turn_longitudinal, turn_transverse);
    }
    transverse = turned_by(damping.transverse[state], angle + turn_transverse, turning);
    longitudinal = turned_by(damping.longitudinal[state], turn_longitudinal, flowing);
}

// One interval's relaxation and exchange over every order, from ``*_in`` into
// ``*_out``, which may be the same planes.
template <typename Real, typename Cplx>
inline void pooled_relax(
    const std::vector<std::vector<Cplx>>& plus_in,
    const std::vector<std::vector<Cplx>>& minus_in,
    const std::vector<std::vector<Cplx>>& longitudinal_in,
    std::vector<std::vector<Cplx>>& plus_out,
    std::vector<std::vector<Cplx>>& minus_out,
    std::vector<std::vector<Cplx>>& longitudinal_out,
    const PoolOperators<Real>& operators,
    const std::vector<Real>& equilibrium,
    const Damping<Real>& damping,
    const Real wout,
    const Real angle,
    const Real flow_rate,
    const Real dt,
    const bool turning,
    const bool flowing,
    std::vector<Cplx>& scratch
) {
    const std::size_t n = longitudinal_in.size();
    const std::size_t m = plus_in.size();
    const std::size_t states = longitudinal_in[0].size();
    for (std::size_t state = 0; state < states; ++state) {
        Cplx unit_transverse{};
        Cplx unit_longitudinal{};
        interval_factors(
            damping, angle, flow_rate, dt, state, turning, flowing,
            unit_transverse, unit_longitudinal
        );
        const Cplx carried = wout * unit_transverse;
        const Cplx conjugated = conjugate(carried);
        for (std::size_t line = 0; line < m; ++line) {
            Cplx forward{};
            Cplx backward{};
            for (std::size_t column = 0; column < m; ++column) {
                const Cplx entry = operators.transverse[line * m + column];
                forward += entry * plus_in[column][state];
                // ``F-`` follows the conjugate of the operator entry by entry,
                // not its transpose: it is the conjugate state.
                backward += conjugate(entry) * minus_in[column][state];
            }
            scratch[line] = forward * carried;
            scratch[m + line] = backward * conjugated;
        }
        for (std::size_t line = 0; line < m; ++line) {
            plus_out[line][state] = scratch[line];
            minus_out[line][state] = scratch[m + line];
        }
        const Cplx spin = wout * unit_longitudinal;
        for (std::size_t line = 0; line < n; ++line) {
            Cplx mixed{};
            for (std::size_t column = 0; column < n; ++column) {
                mixed += operators.longitudinal[line * n + column]
                    * longitudinal_in[column][state];
            }
            scratch[line] = mixed * spin;
        }
        for (std::size_t line = 0; line < n; ++line) {
            longitudinal_out[line][state] = scratch[line];
        }
    }
    // Inflowing spins arrive at equilibrium, so washout scales what the pools
    // held and not what they recover towards.
    for (std::size_t pool = 0; pool < n; ++pool) {
        longitudinal_out[pool][0] +=
            widen(equilibrium[pool] - wout * operators.restored[pool]);
    }
}

template <RfMode MODE>
inline void pair_at(
    const Buffers& buffers, const PoolPass&, const TrainView& view,
    const std::int64_t event, const std::int64_t location, const std::int64_t atom,
    const float theta, Complex& a, Complex& b
) {
    if constexpr (MODE == RfMode::DYNAMIC) {
        dynamic_pair_at(buffers, dynamic_row(buffers, view, event), atom, a, b);
    } else {
        profile_pair(buffers, table_row<MODE>(buffers, event, location), theta, a, b);
    }
}

template <RfMode MODE>
inline void pair_at(
    const Buffers& buffers, const PoolPass& pass, const TrainView& view,
    const std::int64_t event, const std::int64_t location, const std::int64_t atom,
    const DualFloat theta, DualComplex& a, DualComplex& b
) {
    if constexpr (MODE == RfMode::DYNAMIC) {
        dynamic_pair_dual_at(
            buffers, pass.dynamic, dynamic_row(buffers, view, event), atom, a, b
        );
    } else {
        Complex pair_a{};
        Complex pair_b{};
        Complex slope_a{};
        Complex slope_b{};
        profile_pair_slope(
            buffers, table_row<MODE>(buffers, event, location), theta.value,
            pair_a, pair_b, slope_a, slope_b
        );
        a = DualComplex{pair_a, slope_a * theta.tangent};
        b = DualComplex{pair_b, slope_b * theta.tangent};
    }
}

template <RfMode MODE>
inline void pair_slope_at(
    const Buffers& buffers, const PoolPass&, const TrainView& view,
    const std::int64_t event, const std::int64_t location, const std::int64_t atom,
    const float theta, Complex& a, Complex& b, Complex& slope_a, Complex& slope_b
) {
    if constexpr (MODE == RfMode::DYNAMIC) {
        dynamic_pair_at(buffers, dynamic_row(buffers, view, event), atom, a, b);
    } else {
        profile_pair_slope(
            buffers, table_row<MODE>(buffers, event, location), theta, a, b,
            slope_a, slope_b
        );
    }
}

template <RfMode MODE>
inline void pair_slope_at(
    const Buffers& buffers, const PoolPass& pass, const TrainView& view,
    const std::int64_t event, const std::int64_t location, const std::int64_t atom,
    const DualFloat theta, DualComplex& a, DualComplex& b, DualComplex& slope_a,
    DualComplex& slope_b
) {
    if constexpr (MODE == RfMode::DYNAMIC) {
        dynamic_pair_dual_at(
            buffers, pass.dynamic, dynamic_row(buffers, view, event), atom, a, b
        );
    } else {
        profile_pair_slope_dual(
            buffers, table_row<MODE>(buffers, event, location), theta, a, b,
            slope_a, slope_b
        );
    }
}

// Where one work item reads its table, and the directions along its events.
struct PoolItem {
    TrainView view;
    std::int64_t location;
    const float* table;
    const float* direction;
    const std::int32_t* rows;
    const float* dot_duration;
    const float* dot_flip;
    const float* dot_phase;
};

template <RfMode MODE>
inline PoolItem pool_item(
    const Buffers& buffers,
    const PoolPass& pass,
    const PoolShape& shape,
    const std::int64_t work,
    const std::int64_t event_count,
    const std::int64_t output_count
) {
    const TrainView view = train_view(buffers, work, event_count, output_count);
    const std::size_t offset = static_cast<std::size_t>(view.atom) * shape.width;
    auto along = [&](const float* const direction) {
        return direction != nullptr ? direction + view.event_base : nullptr;
    };
    return PoolItem{
        view,
        slice_row<MODE>(buffers, view.atom),
        buffers.pool_values + offset,
        pass.table != nullptr ? pass.table + offset : nullptr,
        buffers.pool_index + view.event_base,
        along(pass.duration),
        along(pass.flip),
        along(pass.phase),
    };
}

// The pulse an event plays, as the voxel receives it.
template <typename Real>
struct PoolPulse {
    std::int64_t transmit;
    Real b1;
    Real alpha;
    Real phi;
};

template <typename Real>
inline PoolPulse<Real> pool_pulse(
    const Buffers& buffers,
    const PoolPass& pass,
    const PoolItem& item,
    const PoolVoxel<Real>& voxel,
    const std::int64_t event
) {
    const std::int64_t atom = item.view.atom;
    const bool shimmed = buffers.shim_count > 1;
    const std::int64_t transmit = shimmed ? transmit_row(buffers, event, atom) : atom;
    const Real b1 = !buffers.transmit
        ? constant<Real>(1.0F)
        : (shimmed ? read_at<Real>(buffers.b1, pass.b1, transmit) : voxel.b1);
    const Real b1_phase = shimmed
        ? read_at<Real>(buffers.b1_phase, pass.b1_phase, transmit)
        : voxel.b1_phase;
    return PoolPulse<Real>{
        transmit,
        b1,
        read_at<Real>(item.view.flip, item.dot_flip, event) * b1,
        read_at<Real>(item.view.phase, item.dot_phase, event) + b1_phase,
    };
}

// The semisolid pool's saturation by a pulse, which reads the bare flip the
// transmit field gives the voxel at the pulse's offset from it.
template <typename Real>
inline Real absorbed_by(
    const Buffers& buffers, const PoolVoxel<Real>& voxel, const Real alpha,
    const std::int64_t event
) {
    const Real offset = constant<Real>(buffers.rf_frequency[event]) - voxel.b0;
    return exponential(
        buffers.saturation[event] * (alpha * alpha * lineshape_at(buffers, offset))
    );
}

// The events of one train after the interval: the shifts, the pulse or the
// reading, and the spoiling. Returns whether the event records, with what the
// coil reads left in ``recorded``.
template <RfMode MODE, typename Real>
inline bool pooled_event(
    const Buffers& buffers,
    const PoolPass& pass,
    const PoolItem& item,
    const PoolVoxel<Real>& voxel,
    const std::int64_t event,
    std::vector<std::vector<typename Carried<Real>::Cplx>>& plus,
    std::vector<std::vector<typename Carried<Real>::Cplx>>& minus,
    std::vector<std::vector<typename Carried<Real>::Cplx>>& longitudinal,
    typename Carried<Real>::Cplx& recorded
) {
    using Cplx = typename Carried<Real>::Cplx;
    const std::size_t m = plus.size();
    const std::size_t n = longitudinal.size();
    const std::uint8_t action = buffers.action[event];
    bool records = false;
    if ((action & PRE_SHIFT) != 0) {
        for (std::size_t pool = 0; pool < m; ++pool) {
            shift(plus[pool], minus[pool]);
        }
    }
    if (buffers.kind[event] == 1) {
        if ((action & INVERSION) != 0) {
            // Every exchanging pool is free water and inverts like it; a
            // semisolid one is saturated by the pulse's own term.
            const Real inverted = constant<Real>(0.0F) - voxel.efficiency;
            for (std::size_t pool = 0; pool < m; ++pool) {
                for (Cplx& value : longitudinal[pool]) {
                    value = value * inverted;
                }
            }
        } else {
            const PoolPulse<Real> pulse =
                pool_pulse(buffers, pass, item, voxel, event);
            if (buffers.pool_semisolid) {
                const Real absorbed = absorbed_by(buffers, voxel, pulse.alpha, event);
                for (Cplx& value : longitudinal[n - 1]) {
                    value = value * absorbed;
                }
            }
            if constexpr (MODE != RfMode::INSTANT) {
                Cplx a{};
                Cplx b{};
                pair_at<MODE>(
                    buffers, pass, item.view, event, item.location, item.view.atom,
                    pulse.alpha, a, b
                );
                const Cplx spun = b
                    * turned_by(
                        constant<Real>(1.0F), constant<Real>(0.0F) - pulse.phi, true
                    );
                for (std::size_t pool = 0; pool < m; ++pool) {
                    rotate_spinor(plus[pool], minus[pool], longitudinal[pool], a, spun);
                }
            } else {
                for (std::size_t pool = 0; pool < m; ++pool) {
                    rotate_by(
                        plus[pool], minus[pool], longitudinal[pool], pulse.alpha,
                        pulse.phi
                    );
                }
            }
        }
    } else if (buffers.kind[event] == 2 && (action & RECORD) != 0) {
        records = true;
        recorded = Cplx{};
        for (std::size_t pool = 0; pool < m; ++pool) {
            recorded += plus[pool][0];
        }
    }
    if ((action & POST_SHIFT) != 0) {
        for (std::size_t pool = 0; pool < m; ++pool) {
            shift(plus[pool], minus[pool]);
        }
    }
    if ((action & SPOIL_AFTER) != 0) {
        for (std::size_t pool = 0; pool < m; ++pool) {
            std::fill(plus[pool].begin(), plus[pool].end(), Cplx{});
            std::fill(minus[pool].begin(), minus[pool].end(), Cplx{});
        }
    } else if ((action & SHIFT_AFTER) != 0) {
        for (std::size_t pool = 0; pool < m; ++pool) {
            shift(plus[pool], minus[pool]);
        }
    }
    return records;
}

inline void emit(const Buffers& buffers, const std::int64_t index, const Complex signal) {
    buffers.output_real[index] = signal.real();
    buffers.output_imag[index] = signal.imag();
}

// A pass carrying a direction writes the signal's derivative along it.
inline void emit(
    const Buffers& buffers, const std::int64_t index, const DualComplex signal
) {
    buffers.output_real[index] = signal.tangent.real();
    buffers.output_imag[index] = signal.tangent.imag();
}

// The forward state machine over tabulated pools, and with a dual scalar the
// derivative of its signal along a direction.
template <RfMode MODE, typename Real>
void simulate_pooled(
    const Buffers& buffers,
    const PoolPass& pass,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count
) {
    using Cplx = typename Carried<Real>::Cplx;
    using Planes = std::vector<std::vector<Cplx>>;
    const PoolShape shape(buffers);
    const std::size_t n = shape.longitudinal;
    const std::size_t m = shape.transverse;
    const bool turning = buffers.off_axis || buffers.moving;
    const bool flowing = buffers.moving;
    const std::size_t states = static_cast<std::size_t>(state_count);
    Planes plus(m, std::vector<Cplx>(states));
    Planes minus(m, std::vector<Cplx>(states));
    Planes longitudinal(n, std::vector<Cplx>(states));
    std::vector<Cplx> scratch(2U * std::max(n, m));
    std::vector<Real> equilibrium(n);
    PoolOperators<Real> operators(shape);
    Damping<Real> damping(states);

    for (std::int64_t work = work_begin; work < work_end; ++work) {
        const PoolItem item =
            pool_item<MODE>(buffers, pass, shape, work, event_count, output_count);
        const std::int64_t atom = item.view.atom;
        const PoolVoxel<Real> voxel(buffers, pass, atom);
        for (std::size_t pool = 0; pool < m; ++pool) {
            std::fill(plus[pool].begin(), plus[pool].end(), Cplx{});
            std::fill(minus[pool].begin(), minus[pool].end(), Cplx{});
        }
        for (std::size_t pool = 0; pool < n; ++pool) {
            equilibrium[pool] = read_at<Real>(item.table, item.direction, pool);
            std::fill(longitudinal[pool].begin(), longitudinal[pool].end(), Cplx{});
            longitudinal[pool][0] = widen(equilibrium[pool]);
        }
        for (std::int64_t event = 0; event < event_count; ++event) {
            const Real dt = read_at<Real>(item.view.duration, item.dot_duration, event);
            damping.set(voxel.damping_rate, dt);
            const Real wout = flowing
                ? washout_out(voxel.washout_rate, dt)
                : constant<Real>(1.0F);
            const Real angle = -2.0F * PI * (voxel.b0 * dt);
            operators.read(
                pool_row<Real>(
                    shape, item.table, item.direction, item.rows[event], 0U,
                    tangent_of(dt)
                ),
                shape
            );
            pooled_relax(
                plus, minus, longitudinal, plus, minus, longitudinal, operators,
                equilibrium, damping, wout, angle, voxel.flow_rate, dt, turning,
                flowing, scratch
            );
            Cplx recorded{};
            if (pooled_event<MODE, Real>(
                    buffers, pass, item, voxel, event, plus, minus, longitudinal,
                    recorded
                )) {
                const Real phase =
                    read_at<Real>(item.view.phase, item.dot_phase, event);
                const Cplx demodulation = turned_by(
                    constant<Real>(1.0F), constant<Real>(0.0F) - phase, true
                );
                emit(
                    buffers,
                    item.view.output_base + buffers.output_index[event],
                    voxel.m0 * (recorded * demodulation)
                );
            }
        }
    }
}

// The adjoint of ``simulate_pooled``: the state entering every event is
// recorded on the way forward and each event replayed from it on the way back.
// With a dual scalar it is the forward-over-reverse pass, the adjoint's
// derivative along a direction.
template <RfMode MODE, typename Real>
void simulate_pooled_adjoint(
    const Buffers& buffers,
    const PoolPass& pass,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count,
    Real* grad_flip_local,
    Real* grad_phase_local,
    Real* grad_duration_local,
    Real* grad_tissue_local
) {
    using Cplx = typename Carried<Real>::Cplx;
    using Plane = std::vector<Cplx>;
    using Planes = std::vector<Plane>;
    const PoolShape shape(buffers);
    const TissueLayout layout(buffers.shim_count);
    const std::int64_t atoms = buffers.atom_count;
    const std::size_t n = shape.longitudinal;
    const std::size_t m = shape.transverse;
    const bool turning = buffers.off_axis || buffers.moving;
    const bool flowing = buffers.moving;
    const std::size_t states = static_cast<std::size_t>(state_count);
    const std::size_t stride = (2U * m + n) * states;
    const Cplx imaginary = seeded<Cplx>(0.0F, 1.0F);
    const Cplx minus_imaginary = seeded<Cplx>(0.0F, -1.0F);

    std::vector<Cplx> trajectory(static_cast<std::size_t>(event_count) * stride);
    Planes plus(m, Plane(states));
    Planes minus(m, Plane(states));
    Planes longitudinal(n, Plane(states));
    Planes plus_bar(m, Plane(states));
    Planes minus_bar(m, Plane(states));
    Planes longitudinal_bar(n, Plane(states));
    Planes plus_relaxed(m, Plane(states));
    Planes minus_relaxed(m, Plane(states));
    Planes longitudinal_relaxed(n, Plane(states));
    Planes plus_shifted(m, Plane(states));
    Planes minus_shifted(m, Plane(states));
    Plane scratch(2U * std::max(n, m));
    Plane mixed_plus(m);
    Plane mixed_minus(m);
    Plane mixed_longitudinal(n);
    std::vector<Real> equilibrium(n);
    std::vector<Real> grad_equilibrium(n);
    std::vector<Real> cotangent(shape.row_width);
    PoolOperators<Real> operators(shape);
    Damping<Real> damping(states);
    const Real zero = constant<Real>(0.0F);

    for (std::int64_t work = work_begin; work < work_end; ++work) {
        const PoolItem item =
            pool_item<MODE>(buffers, pass, shape, work, event_count, output_count);
        const TrainView& view = item.view;
        const std::int64_t atom = view.atom;
        const PoolVoxel<Real> voxel(buffers, pass, atom);
        Real* const grad_flip_train = grad_flip_local + view.event_base;
        Real* const grad_phase_train = grad_phase_local + view.event_base;
        Real* const grad_duration_train = grad_duration_local + view.event_base;
        float* const grad_table = pass.grad_table
            + static_cast<std::size_t>(work) * shape.width;
        float* const grad_table_tangent = pass.grad_table_tangent != nullptr
            ? pass.grad_table_tangent + static_cast<std::size_t>(work) * shape.width
            : nullptr;
        for (std::size_t pool = 0; pool < m; ++pool) {
            std::fill(plus[pool].begin(), plus[pool].end(), Cplx{});
            std::fill(minus[pool].begin(), minus[pool].end(), Cplx{});
        }
        for (std::size_t pool = 0; pool < n; ++pool) {
            equilibrium[pool] = read_at<Real>(item.table, item.direction, pool);
            std::fill(longitudinal[pool].begin(), longitudinal[pool].end(), Cplx{});
            longitudinal[pool][0] = widen(equilibrium[pool]);
        }

        // ---- forward, recording the state entering each event ----
        for (std::int64_t event = 0; event < event_count; ++event) {
            Cplx* slot = trajectory.data() + static_cast<std::size_t>(event) * stride;
            for (std::size_t pool = 0; pool < m; ++pool) {
                std::copy(plus[pool].begin(), plus[pool].end(), slot);
                std::copy(minus[pool].begin(), minus[pool].end(), slot + states);
                slot += 2U * states;
            }
            for (std::size_t pool = 0; pool < n; ++pool) {
                std::copy(longitudinal[pool].begin(), longitudinal[pool].end(), slot);
                slot += states;
            }
            const Real dt = read_at<Real>(view.duration, item.dot_duration, event);
            damping.set(voxel.damping_rate, dt);
            const Real wout = flowing
                ? washout_out(voxel.washout_rate, dt)
                : constant<Real>(1.0F);
            const Real angle = -2.0F * PI * (voxel.b0 * dt);
            operators.read(
                pool_row<Real>(
                    shape, item.table, item.direction, item.rows[event], 0U,
                    tangent_of(dt)
                ),
                shape
            );
            pooled_relax(
                plus, minus, longitudinal, plus, minus, longitudinal, operators,
                equilibrium, damping, wout, angle, voxel.flow_rate, dt, turning,
                flowing, scratch
            );
            Cplx recorded{};
            pooled_event<MODE, Real>(
                buffers, pass, item, voxel, event, plus, minus, longitudinal,
                recorded
            );
        }

        // ---- reverse ----
        for (std::size_t pool = 0; pool < m; ++pool) {
            std::fill(plus_bar[pool].begin(), plus_bar[pool].end(), Cplx{});
            std::fill(minus_bar[pool].begin(), minus_bar[pool].end(), Cplx{});
        }
        for (std::size_t pool = 0; pool < n; ++pool) {
            std::fill(
                longitudinal_bar[pool].begin(), longitudinal_bar[pool].end(), Cplx{}
            );
            grad_equilibrium[pool] = zero;
        }
        Real grad_m0 = zero;
        Real grad_b1 = zero;
        Real grad_b1_phase = zero;
        Real grad_b0 = zero;
        Real grad_efficiency = zero;
        Real grad_damping = zero;
        Real grad_flow = zero;
        Real grad_washout = zero;
        // Transmit gradients are summed per shim: the running pair is flushed
        // to its row whenever the walk back reaches a pulse on a different one.
        std::int64_t held = 0;

        for (std::int64_t event = event_count - 1; event >= 0; --event) {
            const Cplx* slot =
                trajectory.data() + static_cast<std::size_t>(event) * stride;
            for (std::size_t pool = 0; pool < m; ++pool) {
                std::copy(slot, slot + states, plus[pool].begin());
                std::copy(slot + states, slot + 2U * states, minus[pool].begin());
                slot += 2U * states;
            }
            for (std::size_t pool = 0; pool < n; ++pool) {
                std::copy(slot, slot + states, longitudinal[pool].begin());
                slot += states;
            }
            const Real dt = read_at<Real>(view.duration, item.dot_duration, event);
            const float along = tangent_of(dt);
            damping.set(voxel.damping_rate, dt);
            const Real wout = flowing
                ? washout_out(voxel.washout_rate, dt)
                : constant<Real>(1.0F);
            const Real angle = -2.0F * PI * (voxel.b0 * dt);
            const std::int64_t row = item.rows[event];
            operators.read(
                pool_row<Real>(shape, item.table, item.direction, row, 0U, along),
                shape
            );
            const std::uint8_t action = buffers.action[event];

            // replay this event to recover the intra-event states
            pooled_relax(
                plus, minus, longitudinal, plus_relaxed, minus_relaxed,
                longitudinal_relaxed, operators, equilibrium, damping, wout, angle,
                voxel.flow_rate, dt, turning, flowing, scratch
            );
            for (std::size_t pool = 0; pool < m; ++pool) {
                plus_shifted[pool] = plus_relaxed[pool];
                minus_shifted[pool] = minus_relaxed[pool];
                if ((action & PRE_SHIFT) != 0) {
                    shift(plus_shifted[pool], minus_shifted[pool]);
                }
            }

            // --- adjoint of the trailing shift/spoil ---
            for (std::size_t pool = 0; pool < m; ++pool) {
                if ((action & SPOIL_AFTER) != 0) {
                    std::fill(plus_bar[pool].begin(), plus_bar[pool].end(), Cplx{});
                    std::fill(minus_bar[pool].begin(), minus_bar[pool].end(), Cplx{});
                } else if ((action & SHIFT_AFTER) != 0) {
                    shift_adjoint(plus_bar[pool], minus_bar[pool]);
                }
                if ((action & POST_SHIFT) != 0) {
                    shift_adjoint(plus_bar[pool], minus_bar[pool]);
                }
            }

            // --- adjoint of the ADC ---
            if (buffers.kind[event] == 2 && (action & RECORD) != 0) {
                const std::int64_t index =
                    view.output_base + buffers.output_index[event];
                const Cplx seed = seeded<Cplx>(
                    pass.grad_output_real[index], pass.grad_output_imag[index]
                );
                const Real phase = read_at<Real>(view.phase, item.dot_phase, event);
                const Cplx demodulation =
                    turned_by(constant<Real>(1.0F), zero - phase, true);
                // an ADC event carries no RF, so the recorded state is the one
                // left by the pre-shift
                Cplx recorded{};
                for (std::size_t pool = 0; pool < m; ++pool) {
                    recorded += plus_shifted[pool][0];
                }
                grad_m0 += real_part(conjugate(seed) * (recorded * demodulation));
                grad_phase_train[event] += real_part(
                    conjugate(seed)
                    * (voxel.m0 * (recorded * (minus_imaginary * demodulation)))
                );
                const Cplx weighted = conjugate(voxel.m0 * demodulation) * seed;
                for (std::size_t pool = 0; pool < m; ++pool) {
                    plus_bar[pool][0] += weighted;
                }
            }

            // --- adjoint of the RF ---
            if (buffers.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    const Real inverted = zero - voxel.efficiency;
                    for (std::size_t pool = 0; pool < m; ++pool) {
                        for (std::size_t state = 0; state < states; ++state) {
                            Cplx& bar = longitudinal_bar[pool][state];
                            grad_efficiency = grad_efficiency
                                - real_part(
                                    conjugate(bar) * longitudinal_relaxed[pool][state]
                                );
                            bar = bar * inverted;
                        }
                    }
                } else {
                    const PoolPulse<Real> pulse =
                        pool_pulse(buffers, pass, item, voxel, event);
                    const std::int64_t shim =
                        buffers.shim_count > 1 ? buffers.shim_index[event] : 0;
                    if (shim != held) {
                        grad_tissue_local[
                            (layout.base[B1_INDEX] + held) * atoms + atom
                        ] += grad_b1;
                        grad_tissue_local[
                            (layout.base[B1_PHASE_INDEX] + held) * atoms + atom
                        ] += grad_b1_phase;
                        grad_b1 = zero;
                        grad_b1_phase = zero;
                        held = shim;
                    }
                    Real grad_alpha = zero;
                    Real grad_phi = zero;
                    if constexpr (MODE != RfMode::INSTANT) {
                        Cplx pair_a{};
                        Cplx pair_b{};
                        Cplx slope_a{};
                        Cplx slope_b{};
                        pair_slope_at<MODE>(
                            buffers, pass, view, event, item.location, atom,
                            pulse.alpha, pair_a, pair_b, slope_a, slope_b
                        );
                        const Cplx turn =
                            turned_by(constant<Real>(1.0F), zero - pulse.phi, true);
                        const Cplx spun = pair_b * turn;
                        Cplx grad_a{};
                        Cplx grad_b{};
                        for (std::size_t pool = 0; pool < m; ++pool) {
                            rotate_adjoint_spinor_by(
                                plus_shifted[pool], minus_shifted[pool],
                                longitudinal_relaxed[pool], plus_bar[pool],
                                minus_bar[pool], longitudinal_bar[pool], pair_a, spun,
                                grad_a, grad_b
                            );
                        }
                        // The RF phase turns the axis once the pair is out,
                        // so it reaches ``b`` alone -- under either mode.
                        grad_phi = real_part(conjugate(grad_b) * (minus_imaginary * spun));
                        if constexpr (MODE == RfMode::DYNAMIC) {
                            // The flip is inside the pair rather than read
                            // against it, so the cotangent goes out on the pair.
                            const std::int64_t at = dynamic_offset(
                                buffers, dynamic_row(buffers, view, event), atom
                            );
                            // ``b`` was turned by the phase after the pair came
                            // out, so the cotangent turns back the other way.
                            const Cplx unturned = grad_b * conjugate(turn);
                            const Real parts[4] = {
                                real_part(grad_a), imag_part(grad_a),
                                real_part(unturned), imag_part(unturned),
                            };
                            for (std::size_t part = 0; part < 4; ++part) {
                                deposit(
                                    pass.grad_dynamic + at + part,
                                    pass.grad_dynamic_tangent != nullptr
                                        ? pass.grad_dynamic_tangent + at + part
                                        : nullptr,
                                    parts[part]
                                );
                            }
                        } else {
                            grad_alpha = real_part(conjugate(grad_a) * slope_a)
                                + real_part(conjugate(grad_b) * (slope_b * turn));
                        }
                    } else {
                        for (std::size_t pool = 0; pool < m; ++pool) {
                            rotate_adjoint_by(
                                plus_shifted[pool], minus_shifted[pool],
                                longitudinal_relaxed[pool], plus_bar[pool],
                                minus_bar[pool], longitudinal_bar[pool], pulse.alpha,
                                pulse.phi, grad_alpha, grad_phi
                            );
                        }
                    }
                    if (buffers.pool_semisolid) {
                        // The pulse scales every order of the semisolid pool by
                        // one real number, so its cotangent is a single sum
                        // over the states it multiplied.
                        const Real offset =
                            constant<Real>(buffers.rf_frequency[event]) - voxel.b0;
                        Real shape_value = zero;
                        Real shape_slope = zero;
                        lineshape_at_slope(buffers, offset, shape_value, shape_slope);
                        const float deposited = buffers.saturation[event];
                        const Real absorbed = exponential(
                            deposited * (pulse.alpha * pulse.alpha * shape_value)
                        );
                        Real grad_absorbed = zero;
                        Plane& bar = longitudinal_bar[n - 1];
                        for (std::size_t state = 0; state < states; ++state) {
                            grad_absorbed += real_part(
                                conjugate(bar[state])
                                * longitudinal_relaxed[n - 1][state]
                            );
                            bar[state] = bar[state] * absorbed;
                        }
                        const Real grad_exponent = grad_absorbed * absorbed;
                        grad_alpha += (2.0F * deposited)
                            * (grad_exponent * pulse.alpha * shape_value);
                        // The lineshape is read at the pulse's offset from the
                        // voxel, so a step in the voxel's own off-resonance
                        // moves the read the other way.
                        grad_b0 = grad_b0
                            - deposited
                                * (grad_exponent * pulse.alpha * pulse.alpha
                                   * shape_slope);
                    }
                    grad_flip_train[event] += grad_alpha * pulse.b1;
                    grad_b1 += grad_alpha
                        * read_at<Real>(view.flip, item.dot_flip, event);
                    grad_phase_train[event] += grad_phi;
                    grad_b1_phase += grad_phi;
                }
            }

            if ((action & PRE_SHIFT) != 0) {
                for (std::size_t pool = 0; pool < m; ++pool) {
                    shift_adjoint(plus_bar[pool], minus_bar[pool]);
                }
            }

            // --- adjoint of the interval ---
            for (Real& entry : cotangent) {
                entry = zero;
            }
            Real grad_wout = zero;
            // Order zero also carries the recovery, which is the equilibrium
            // less what washout leaves of the operator applied to it. Read
            // before the sweep below replaces the seeds.
            for (std::size_t pool = 0; pool < n; ++pool) {
                const Real seed = real_part(longitudinal_bar[pool][0]);
                grad_equilibrium[pool] += seed;
                cotangent[shape.restored() + pool] += zero - wout * seed;
                grad_wout = grad_wout - seed * operators.restored[pool];
            }
            Real grad_angle = zero;
            // The b-factor reaches every order through a different weight,
            // and flow through a turn of its own, so both collect weighted
            // sums rather than scalar ones.
            Real grad_b_factor = zero;
            Real grad_turn = zero;
            for (std::size_t state = 0; state < states; ++state) {
                Cplx unit_transverse{};
                Cplx unit_longitudinal{};
                interval_factors(
                    damping, angle, voxel.flow_rate, dt, state, turning, flowing,
                    unit_transverse, unit_longitudinal
                );
                const Cplx carried = wout * unit_transverse;
                const Cplx spin = wout * unit_longitudinal;
                for (std::size_t line = 0; line < m; ++line) {
                    Cplx forward{};
                    Cplx backward{};
                    for (std::size_t column = 0; column < m; ++column) {
                        const Cplx entry = operators.transverse[line * m + column];
                        forward += entry * plus[column][state];
                        backward += conjugate(entry) * minus[column][state];
                    }
                    mixed_plus[line] = forward;
                    mixed_minus[line] = backward;
                }
                for (std::size_t line = 0; line < n; ++line) {
                    Cplx mixed{};
                    for (std::size_t column = 0; column < n; ++column) {
                        mixed += operators.longitudinal[line * n + column]
                            * longitudinal[column][state];
                    }
                    mixed_longitudinal[line] = mixed;
                }
                // The damping is homogeneous of degree one in every state it
                // acts on, so its gradient times the damping itself is the
                // cotangent taken against the states the interval leaves; the
                // turns are the same derivatives with an imaginary weight.
                Real transverse_scaled = zero;
                Real transverse_angle = zero;
                for (std::size_t line = 0; line < m; ++line) {
                    const Cplx seed_plus = conjugate(plus_bar[line][state]);
                    const Cplx seed_minus = conjugate(minus_bar[line][state]);
                    const Cplx out_plus = seed_plus * (carried * mixed_plus[line]);
                    const Cplx out_minus =
                        seed_minus * (conjugate(carried) * mixed_minus[line]);
                    transverse_scaled += real_part(out_plus + out_minus);
                    transverse_angle += real_part(imaginary * (out_plus - out_minus));
                    grad_wout += real_part(
                        seed_plus * (unit_transverse * mixed_plus[line])
                        + seed_minus * (conjugate(unit_transverse) * mixed_minus[line])
                    );
                }
                Real longitudinal_scaled = zero;
                Real longitudinal_angle = zero;
                for (std::size_t line = 0; line < n; ++line) {
                    const Cplx seed = conjugate(longitudinal_bar[line][state]);
                    const Cplx out = seed * (spin * mixed_longitudinal[line]);
                    longitudinal_scaled += real_part(out);
                    longitudinal_angle += real_part(imaginary * out);
                    grad_wout += real_part(
                        seed * (unit_longitudinal * mixed_longitudinal[line])
                    );
                }
                // The operators' own entries. ``F-`` follows the conjugate of
                // the transverse operator, so its cotangent lands on the entry
                // itself rather than on the conjugate of it.
                const Cplx released = conjugate(carried);
                for (std::size_t line = 0; line < m; ++line) {
                    const Cplx bar_plus = plus_bar[line][state];
                    const Cplx seed_minus = conjugate(minus_bar[line][state]);
                    for (std::size_t column = 0; column < m; ++column) {
                        const Cplx entry = released
                            * (bar_plus * conjugate(plus[column][state])
                               + seed_minus * minus[column][state]);
                        const std::size_t at = shape.across() + 2U * (line * m + column);
                        cotangent[at] += real_part(entry);
                        cotangent[at + 1U] += imag_part(entry);
                    }
                }
                for (std::size_t line = 0; line < n; ++line) {
                    const Cplx seed = conjugate(longitudinal_bar[line][state]) * spin;
                    for (std::size_t column = 0; column < n; ++column) {
                        cotangent[line * n + column] +=
                            real_part(seed * longitudinal[column][state]);
                    }
                }
                // The seeds back through the interval.
                for (std::size_t column = 0; column < m; ++column) {
                    Cplx back_plus{};
                    Cplx back_minus{};
                    for (std::size_t line = 0; line < m; ++line) {
                        const Cplx step =
                            carried * operators.transverse[line * m + column];
                        back_plus += conjugate(step) * plus_bar[line][state];
                        back_minus += step * minus_bar[line][state];
                    }
                    scratch[column] = back_plus;
                    scratch[m + column] = back_minus;
                }
                for (std::size_t column = 0; column < m; ++column) {
                    plus_bar[column][state] = scratch[column];
                    minus_bar[column][state] = scratch[m + column];
                }
                const Cplx returned = conjugate(spin);
                for (std::size_t column = 0; column < n; ++column) {
                    Cplx back{};
                    for (std::size_t line = 0; line < n; ++line) {
                        back += operators.longitudinal[line * n + column]
                            * (returned * longitudinal_bar[line][state]);
                    }
                    scratch[column] = back;
                }
                for (std::size_t column = 0; column < n; ++column) {
                    longitudinal_bar[column][state] = scratch[column];
                }
                const float order = static_cast<float>(state);
                grad_angle += transverse_angle;
                grad_b_factor = grad_b_factor
                    - (transverse_weight(state) * transverse_scaled
                       + longitudinal_weight(state) * longitudinal_scaled);
                grad_turn = grad_turn
                    - ((order + 0.5F) * transverse_angle + order * longitudinal_angle);
            }

            // The row this event read is shared with every event of its
            // length; its cotangent is summed into it, and reaches the event's
            // own length through the row's slope.
            Real grad_table_duration = zero;
            float* const grad_row = grad_table + shape.row(row, 0U);
            float* const grad_row_tangent = grad_table_tangent != nullptr
                ? grad_table_tangent + shape.row(row, 0U)
                : nullptr;
            for (std::size_t entry = 0; entry < shape.row_width; ++entry) {
                deposit(
                    grad_row + entry,
                    grad_row_tangent != nullptr ? grad_row_tangent + entry : nullptr,
                    cotangent[entry]
                );
            }
            if (shape.blocks > 1U) {
                const PoolRow<Real> slope = pool_row<Real>(
                    shape, item.table, item.direction, row, 1U, along
                );
                for (std::size_t entry = 0; entry < shape.row_width; ++entry) {
                    grad_table_duration += cotangent[entry] * slope.real(entry);
                }
                if (grad_table_tangent != nullptr) {
                    float* const grad_slope = grad_table_tangent + shape.row(row, 1U);
                    for (std::size_t entry = 0; entry < shape.row_width; ++entry) {
                        deposit_along(grad_slope + entry, cotangent[entry], along);
                    }
                }
            }

            // Washout scales every factor the interval applies and the
            // recovery it leaves; past the clamp nothing depends on the rate.
            const Real grad_fraction =
                flowing && primal(voxel.washout_rate * dt) < 1.0F
                    ? zero - grad_wout
                    : zero;
            grad_b0 += grad_angle * (-2.0F * PI * dt);
            grad_damping += grad_b_factor * dt;
            grad_flow += grad_turn * dt;
            grad_washout += grad_fraction * dt;
            grad_duration_train[event] += grad_angle * (-2.0F * PI * voxel.b0)
                + grad_b_factor * voxel.damping_rate + grad_turn * voxel.flow_rate
                + grad_fraction * voxel.washout_rate + grad_table_duration;
        }

        // The equilibrium is also where every pool starts, which the walk back
        // reaches last.
        for (std::size_t pool = 0; pool < n; ++pool) {
            grad_equilibrium[pool] += real_part(longitudinal_bar[pool][0]);
            deposit(
                grad_table + pool,
                grad_table_tangent != nullptr ? grad_table_tangent + pool : nullptr,
                grad_equilibrium[pool]
            );
        }
        grad_tissue_local[layout.base[2] * atoms + atom] += grad_m0;
        grad_tissue_local[(layout.base[B1_INDEX] + held) * atoms + atom] += grad_b1;
        grad_tissue_local[(layout.base[B1_PHASE_INDEX] + held) * atoms + atom] +=
            grad_b1_phase;
        grad_tissue_local[layout.base[5] * atoms + atom] += grad_b0;
        grad_tissue_local[layout.base[6] * atoms + atom] += grad_efficiency;
        grad_tissue_local[layout.base[7] * atoms + atom] += grad_damping;
        // One buffer drives two rates, so the velocity gradient is the sum of
        // what each geometry carries back.
        grad_tissue_local[layout.base[8] * atoms + atom] +=
            buffers.flow_scale * grad_flow
            + (speed_direction(primal(voxel.velocity)) * buffers.washout_scale)
                * grad_washout;
    }
}

// What each pass hands the two bodies.
inline PoolPass pool_pass(const JvpBuffers& buffers) {
    PoolPass pass;
    pass.m0 = buffers.m0;
    pass.b1 = buffers.b1;
    pass.b1_phase = buffers.b1_phase;
    pass.b0 = buffers.b0;
    pass.inversion_efficiency = buffers.inversion_efficiency;
    pass.diffusion = buffers.diffusion;
    pass.velocity = buffers.velocity;
    pass.duration = buffers.duration;
    pass.flip = buffers.flip;
    pass.phase = buffers.phase;
    pass.dynamic = buffers.dynamic;
    pass.table = buffers.pool_values;
    return pass;
}

inline PoolPass pool_pass(const VjpBuffers& buffers) {
    PoolPass pass;
    pass.grad_output_real = buffers.grad_output_real;
    pass.grad_output_imag = buffers.grad_output_imag;
    pass.grad_table = buffers.grad_pool;
    pass.grad_dynamic = buffers.grad_dynamic;
    return pass;
}

inline PoolPass pool_pass(const VjpJvpBuffers& buffers) {
    PoolPass pass;
    pass.m0 = buffers.dot_m0;
    pass.b1 = buffers.dot_b1;
    pass.b1_phase = buffers.dot_b1_phase;
    pass.b0 = buffers.dot_b0;
    pass.inversion_efficiency = buffers.dot_inversion_efficiency;
    pass.diffusion = buffers.dot_diffusion;
    pass.velocity = buffers.dot_velocity;
    pass.duration = buffers.dot_duration;
    pass.flip = buffers.dot_flip;
    pass.phase = buffers.dot_phase;
    pass.dynamic = buffers.dynamic;
    pass.table = buffers.pool_values;
    pass.grad_output_real = buffers.grad_output_real;
    pass.grad_output_imag = buffers.grad_output_imag;
    pass.grad_table = buffers.grad_dot_pool;
    pass.grad_table_tangent = buffers.grad_pool;
    pass.grad_dynamic = buffers.grad_dot_dynamic;
    pass.grad_dynamic_tangent = buffers.grad_dynamic;
    return pass;
}

template <RfMode MODE>
void simulate_pooled_range(
    const Buffers& buffers,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count
) {
    simulate_pooled<MODE, float>(
        buffers, PoolPass{}, work_begin, work_end, event_count, state_count,
        output_count
    );
}

template <RfMode MODE>
void simulate_pooled_jvp_range(
    const JvpBuffers& buffers,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count
) {
    simulate_pooled<MODE, DualFloat>(
        buffers.primal, pool_pass(buffers), work_begin, work_end, event_count,
        state_count, output_count
    );
}

template <RfMode MODE>
void simulate_pooled_vjp_range(
    const VjpBuffers& buffers,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count,
    float* grad_flip_local,
    float* grad_phase_local,
    float* grad_duration_local,
    float* grad_tissue_local
) {
    simulate_pooled_adjoint<MODE, float>(
        buffers.primal, pool_pass(buffers), work_begin, work_end, event_count,
        state_count, output_count, grad_flip_local, grad_phase_local,
        grad_duration_local, grad_tissue_local
    );
}

template <RfMode MODE>
void simulate_pooled_vjp_jvp_range(
    const VjpJvpBuffers& buffers,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count,
    DualFloat* grad_flip_local,
    DualFloat* grad_phase_local,
    DualFloat* grad_duration_local,
    DualFloat* grad_tissue_local
) {
    simulate_pooled_adjoint<MODE, DualFloat>(
        buffers.primal, pool_pass(buffers), work_begin, work_end, event_count,
        state_count, output_count, grad_flip_local, grad_phase_local,
        grad_duration_local, grad_tissue_local
    );
}

// Vector clones of the single-pool derivative kernels.
//
// ``target_clones`` multiplies a function by its instruction sets and the pool
// flag multiplies the instantiations, so cloning all eight of each takes the
// object past twice the size it had with one pool. The two-pool step is a 2x2
// matvec once per interval, which the wider registers do not shorten, so the
// clones are spent on the single-pool instantiations instead -- whose scalar
// state loop they do.
//
// A clone is a copy of the *caller*, so the body has to reach it by inlining;
// that is what the ``always_inline`` on the range kernels is for. Taking the
// address of one still emits an out-of-line copy, which is what the two-pool
// dispatch selects.
#if TORCHSIM_MULTIVERSIONED
#define CLONED_KERNEL \
    __attribute__((target_clones("default", "sse4.2", "avx2", "avx512f")))
#else
#define CLONED_KERNEL
#endif

template <RfMode MODE>
CLONED_KERNEL
void simulate_jvp_single_pool(
    const JvpBuffers& buffers,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count
) {
    simulate_jvp_range<MODE, Pools::ONE>(
        buffers, work_begin, work_end, event_count, state_count, output_count
    );
}

template <RfMode MODE>
CLONED_KERNEL
void simulate_vjp_single_pool(
    const VjpBuffers& buffers,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count,
    float* grad_flip_local,
    float* grad_phase_local,
    float* grad_duration_local,
    float* grad_tissue_local
) {
    simulate_vjp_range<MODE, Pools::ONE>(
        buffers, work_begin, work_end, event_count, state_count, output_count,
        grad_flip_local, grad_phase_local, grad_duration_local,
        grad_tissue_local
    );
}

// The forward-over-reverse body is the largest of the four and the rarest
// path, so it is the one whose clones are given up once a fourth pool count
// takes the object past the size a build should carry. The three others keep
// theirs.
template <RfMode MODE>
void simulate_vjp_jvp_single_pool(
    const VjpJvpBuffers& buffers,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count,
    DualFloat* grad_flip_local,
    DualFloat* grad_phase_local,
    DualFloat* grad_duration_local,
    DualFloat* grad_tissue_local
) {
    simulate_vjp_jvp_range<MODE, Pools::ONE>(
        buffers, work_begin, work_end, event_count, state_count, output_count,
        grad_flip_local, grad_phase_local, grad_duration_local,
        grad_tissue_local
    );
}

// The packed pointers arrive in the order the parameter registry declares:
// every tissue property, then duration, kind, flip, phase, action and the
// output index. Naming them once here means a new parameter moves one table
// rather than every entry point's aggregate initializer.
inline Buffers packed_buffers(
    void* const* raw,
    float* const output_real,
    float* const output_imag,
    const std::int64_t atom_count,
    const std::int64_t train_count,
    const float flow_scale,
    const float washout_scale,
    const std::int64_t shim_count,
    const void* const profile = nullptr,
    const void* const profile_index = nullptr,
    const std::int64_t profile_bins = 0,
    const std::int64_t locations = 1,
    const float profile_step = 1.0F,
    const void* const dynamic = nullptr,
    const void* const dynamic_index = nullptr,
    const void* const lineshape = nullptr,
    const std::int64_t lineshape_bins = 0,
    const float lineshape_step = 1.0F,
    const int features = FEATURE_ALL
) {
    Buffers buffers{};
    buffers.off_axis = (features & FEATURE_OFF_AXIS) != 0;
    buffers.moving = (features & FEATURE_MOVING) != 0;
    buffers.diffusing = (features & FEATURE_DIFFUSING) != 0;
    buffers.transmit = (features & FEATURE_TRANSMIT) != 0;
    buffers.density = (features & FEATURE_DENSITY) != 0;
    buffers.inverting = (features & FEATURE_INVERTING) != 0;
    const float** tissue[TISSUE_COUNT] = {
        &buffers.t1, &buffers.t2, &buffers.m0, &buffers.b1,
        &buffers.b1_phase, &buffers.b0, &buffers.inversion_efficiency,
        &buffers.diffusion, &buffers.velocity,
        &buffers.bound_fraction, &buffers.bound_exchange, &buffers.t1_bound,
        &buffers.pool_b_fraction, &buffers.pool_b_exchange,
        &buffers.t1_pool_b, &buffers.t2_pool_b, &buffers.pool_b_shift,
    };
    static_assert(std::size(tissue) == TISSUE_COUNT, "tissue pointers");
    for (std::size_t index = 0; index < TISSUE_COUNT; ++index) {
        *tissue[index] = static_cast<const float*>(raw[index]);
    }
    buffers.duration = static_cast<const float*>(raw[TISSUE_COUNT]);
    buffers.kind = static_cast<const std::int32_t*>(raw[TISSUE_COUNT + 1]);
    buffers.flip = static_cast<const float*>(raw[TISSUE_COUNT + 2]);
    buffers.phase = static_cast<const float*>(raw[TISSUE_COUNT + 3]);
    buffers.action = static_cast<const std::uint8_t*>(raw[TISSUE_COUNT + 4]);
    buffers.output_index =
        static_cast<const std::int32_t*>(raw[TISSUE_COUNT + 5]);
    buffers.shim_index = static_cast<const std::int32_t*>(raw[TISSUE_COUNT + 6]);
    buffers.saturation = static_cast<const float*>(raw[TISSUE_COUNT + 7]);
    buffers.rf_frequency = static_cast<const float*>(raw[TISSUE_COUNT + 8]);
    buffers.output_real = output_real;
    buffers.output_imag = output_imag;
    buffers.atom_count = atom_count;
    buffers.train_count = train_count;
    buffers.flow_scale = flow_scale;
    buffers.washout_scale = washout_scale;
    buffers.shim_count = shim_count;
    buffers.profile = static_cast<const float*>(profile);
    buffers.profile_index = static_cast<const std::int32_t*>(profile_index);
    buffers.profile_bins = profile_bins;
    buffers.locations = locations;
    buffers.profile_step = profile_step;
    buffers.dynamic = static_cast<const float*>(dynamic);
    buffers.dynamic_index = static_cast<const std::int32_t*>(dynamic_index);
    buffers.lineshape = static_cast<const float*>(lineshape);
    buffers.lineshape_bins = lineshape_bins;
    buffers.lineshape_step = lineshape_step;
    return buffers;
}

// Every entry point's pointer sequence ends with the same thirteen slots, so
// they are named and counted back from the end rather than written out at each
// site. An entry point reads only the ones its kernel can use; the rest are
// null. Naming them is what makes a slot left unassigned a visible omission
// instead of a null the kernel writes through.
enum class Tail : Py_ssize_t {
    POOLS = 13,
    POOL_INDEX = 12,
    POOL_DIRECTION = 11,
    POOL_GRADIENT = 10,
    POOL_CURVATURE = 9,
    TABLE = 8,
    TABLE_INDEX = 7,
    PAIRS = 6,
    PAIR_INDEX = 5,
    DIRECTION = 4,
    GRADIENT = 3,
    CURVATURE = 2,
    ABSORPTION = 1,
};

constexpr Py_ssize_t TAIL_COUNT = static_cast<Py_ssize_t>(Tail::POOLS);

inline void* tail_slot(
    void* const* const raw, const Py_ssize_t expected, const Tail slot
) {
    return raw[expected - static_cast<Py_ssize_t>(slot)];
}

// Pool kinds past the closed forms: exchanging pools tabulated per interval,
// without and with a semisolid pool beside them.
constexpr int POOLS_TABULATED = 4;
constexpr int POOLS_TABULATED_SATURATED = 5;

inline bool tabulated(const int pool_kind) {
    return pool_kind == POOLS_TABULATED || pool_kind == POOLS_TABULATED_SATURATED;
}

// Hands a tabulated run's table and its row index to the buffers, and checks
// the counts that size them. Returns false with a Python error set.
inline bool attach_pools(
    Buffers& buffers,
    void* const* const raw,
    const Py_ssize_t expected,
    const int pool_kind,
    const long long pool_count,
    const long long pool_rows,
    const long long pool_blocks
) {
    if (!tabulated(pool_kind)) {
        return true;
    }
    buffers.pool_values =
        static_cast<const float*>(tail_slot(raw, expected, Tail::POOLS));
    buffers.pool_index =
        static_cast<const std::int32_t*>(tail_slot(raw, expected, Tail::POOL_INDEX));
    buffers.pool_count = static_cast<std::int64_t>(pool_count);
    buffers.pool_rows = static_cast<std::int64_t>(pool_rows);
    buffers.pool_blocks = static_cast<std::int64_t>(pool_blocks);
    buffers.pool_semisolid = pool_kind == POOLS_TABULATED_SATURATED;
    if (buffers.pool_values == nullptr || buffers.pool_index == nullptr
        || pool_count < 1 || pool_rows < 1 || pool_blocks < 1
        || (buffers.pool_semisolid && buffers.lineshape == nullptr)) {
        PyErr_SetString(PyExc_ValueError, "invalid pool table");
        return false;
    }
    return true;
}

inline Pools pool_of(const int pool_kind) {
    if (tabulated(pool_kind)) {
        return Pools::TABULATED;
    }
    return pool_kind == 3
        ? Pools::THREE
        : (pool_kind == 2
            ? Pools::EXCHANGING
            : (pool_kind == 1 ? Pools::SEMISOLID : Pools::ONE));
}

bool parse_pointer(PyObject* sequence, const Py_ssize_t index, void** pointer) {
    PyObject* value = PySequence_GetItem(sequence, index);
    if (value == nullptr) {
        return false;
    }
    *pointer = PyLong_AsVoidPtr(value);
    Py_DECREF(value);
    return *pointer != nullptr || !PyErr_Occurred();
}


PyObject* simulate(PyObject*, PyObject* arguments) {
    PyObject* pointers = nullptr;
    long long atom_count = 0;
    long long train_count = 0;
    long long event_count = 0;
    long long state_count = 0;
    long long output_count = 0;
    int requested_threads = 0;
    int real_axis = -1;
    double flow_scale = 0.0;
    double washout_scale = 0.0;
    long long shim_count = 1;
    long long locations = 1;
    long long profile_bins = 0;
    double profile_step = 1.0;
    long long lineshape_bins = 0;
    double lineshape_step = 1.0;
    int pool_kind = 0;
    int features = FEATURE_ALL;
    long long pool_count = 0;
    long long pool_rows = 0;
    long long pool_blocks = 0;
    if (!PyArg_ParseTuple(
            arguments,
            "OLLLLLiiddLLLdLdiiLLL",
            &pointers,
            &atom_count,
            &train_count,
            &event_count,
            &state_count,
            &output_count,
            &requested_threads,
            &real_axis,
            &flow_scale,
            &washout_scale,
            &shim_count,
            &locations,
            &profile_bins,
            &profile_step,
            &lineshape_bins,
            &lineshape_step,
            &pool_kind,
            &features,
            &pool_count,
            &pool_rows,
            &pool_blocks
        )) {
        return nullptr;
    }
    const Pools pools = pool_of(pool_kind);
    // The packed buffers, the two output planes, then the transition tables
    // and the per-event index that says which an event reads, then the
    // per-voxel rotations, their own index and a direction along them, then
    // the bound pool's lineshape -- all null when the sequence has none.
    constexpr Py_ssize_t expected = PACKED_COUNT + 2 + TAIL_COUNT;
    if (!PySequence_Check(pointers) || PySequence_Size(pointers) != expected) {
        PyErr_SetString(PyExc_ValueError, "wrong number of buffer pointers");
        return nullptr;
    }
    if (atom_count < 0 || train_count < 1 || event_count < 0 || state_count < 1
        || output_count < 0) {
        PyErr_SetString(PyExc_ValueError, "invalid EPG buffer dimensions");
        return nullptr;
    }

    void* raw[expected]{};
    for (Py_ssize_t index = 0; index < expected; ++index) {
        if (!parse_pointer(pointers, index, &raw[index])) {
            return nullptr;
        }
    }
    Buffers buffers = packed_buffers(
        raw,
        static_cast<float*>(raw[PACKED_COUNT]),
        static_cast<float*>(raw[PACKED_COUNT + 1]),
        static_cast<std::int64_t>(atom_count),
        static_cast<std::int64_t>(train_count),
        static_cast<float>(flow_scale),
        static_cast<float>(washout_scale),
        static_cast<std::int64_t>(shim_count),
        tail_slot(raw, expected, Tail::TABLE),
        tail_slot(raw, expected, Tail::TABLE_INDEX),
        static_cast<std::int64_t>(profile_bins),
        static_cast<std::int64_t>(locations),
        static_cast<float>(profile_step),
        tail_slot(raw, expected, Tail::PAIRS),
        tail_slot(raw, expected, Tail::PAIR_INDEX),
        tail_slot(raw, expected, Tail::ABSORPTION),
        static_cast<std::int64_t>(lineshape_bins),
        static_cast<float>(lineshape_step),
        features
    );
    if (!attach_pools(
            buffers, raw, expected, pool_kind, pool_count, pool_rows, pool_blocks
        )) {
        return nullptr;
    }

    // TORCHSIM_LANES=1 selects a lane-vectorized forward that walks the
    // (atom, train block) product instead of (atom, train), putting a block of
    // trains in the SIMD lanes. It is opt-in: it is measurably faster only for
    // very wide batches, and it does not agree bit for bit with the scalar
    // kernel, so enabling it changes results in the last place.
    const char* const lane_override = std::getenv("TORCHSIM_LANES");
    const bool lanes_enabled = lane_override != nullptr && lane_override[0] == '1';
    const bool vectorize = lanes_enabled && train_count >= 4
        && buffers.shim_count == 1
        && buffers.profile == nullptr
        // A per-voxel pair is the pulse, not a refinement of it: the lane
        // kernel carries no rotation to read one into, so it would play a hard
        // pulse instead of the one the caller described.
        && buffers.dynamic == nullptr
        && pools == Pools::ONE
        && !any_diffusion(buffers.diffusion, buffers.atom_count)
        && !any_diffusion(buffers.velocity, buffers.atom_count);
    const std::int64_t lane_blocks =
        (static_cast<std::int64_t>(train_count) + static_cast<std::int64_t>(LANES) - 1)
        / static_cast<std::int64_t>(LANES);
    const std::int64_t work_count = static_cast<std::int64_t>(atom_count)
        * (vectorize ? lane_blocks : static_cast<std::int64_t>(train_count));
    void (*kernel)(
        const Buffers&, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
        std::int64_t
    ) = vectorize
        ? &simulate_lane_range
        : &simulate_range<RfMode::INSTANT, Pools::ONE>;
    if (!vectorize) {
        const RfMode mode = buffers.dynamic != nullptr
            ? RfMode::DYNAMIC
            : (buffers.profile != nullptr ? RfMode::PROFILED
                                              : RfMode::INSTANT);
        if (pools == Pools::THREE) {
            kernel = mode == RfMode::DYNAMIC
                ? &simulate_range<RfMode::DYNAMIC, Pools::THREE>
                : (mode == RfMode::PROFILED
                    ? &simulate_range<RfMode::PROFILED, Pools::THREE>
                    : &simulate_range<RfMode::INSTANT, Pools::THREE>);
        } else if (pools == Pools::EXCHANGING) {
            kernel = mode == RfMode::DYNAMIC
                ? &simulate_range<RfMode::DYNAMIC, Pools::EXCHANGING>
                : (mode == RfMode::PROFILED
                    ? &simulate_range<RfMode::PROFILED, Pools::EXCHANGING>
                    : &simulate_range<RfMode::INSTANT, Pools::EXCHANGING>);
        } else if (pools == Pools::SEMISOLID) {
            kernel = mode == RfMode::DYNAMIC
                ? &simulate_range<RfMode::DYNAMIC, Pools::SEMISOLID>
                : (mode == RfMode::PROFILED
                    ? &simulate_range<RfMode::PROFILED, Pools::SEMISOLID>
                    : &simulate_range<RfMode::INSTANT, Pools::SEMISOLID>);
        } else if (pools == Pools::TABULATED) {
            kernel = mode == RfMode::DYNAMIC
                ? &simulate_pooled_range<RfMode::DYNAMIC>
                : (mode == RfMode::PROFILED
                    ? &simulate_pooled_range<RfMode::PROFILED>
                    : &simulate_pooled_range<RfMode::INSTANT>);
        } else if (mode == RfMode::DYNAMIC) {
            kernel = &simulate_range<RfMode::DYNAMIC, Pools::ONE>;
        } else if (mode == RfMode::PROFILED) {
            kernel = &simulate_range<RfMode::PROFILED, Pools::ONE>;
        }
    }
    // The caller establishes the real-subspace conditions; axis 1 puts the
    // signal on the imaginary axis, which is the representation below.
    // A bound pool leaves that subspace: saturation is real, but the pool is
    // reached by an operator the real kernel does not carry.
    if (real_axis == 1 && pools == Pools::ONE) {
        kernel = &simulate_real_range;
    }
    const unsigned int thread_count = worker_count(requested_threads, work_count);

    Py_BEGIN_ALLOW_THREADS
    {
        const std::int64_t block = (work_count + thread_count - 1) / thread_count;
        WorkerPool::instance().run(thread_count, [&](const unsigned int slot) {
            const std::int64_t begin = static_cast<std::int64_t>(slot) * block;
            const std::int64_t end = std::min<std::int64_t>(work_count, begin + block);
            if (begin < end) {
                kernel(
                    buffers, begin, end, event_count, state_count, output_count
                );
            }
        });
    }
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

// Spread one forward-mode pass over the pool. Shared by the standalone entry
// point and by the fused optimizer, so both pick the same kernel for a given
// subspace verdict.
void dispatch_jvp(
    const JvpBuffers& buffers,
    const std::int64_t atom_count,
    const std::int64_t train_count,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count,
    const int requested_threads,
    const int real_axis,
    const Pools pools
) {
    const bool single = pools == Pools::ONE;
    // A block of lanes has to be filled from one axis or the other, and the one
    // to fill it from is whichever the run is wide in. A dictionary is wide in
    // atoms and has one train; a slice-profiled run is the other way about.
    // Filling a block from an axis with fewer than eight entries pads the rest
    // with repeats and pays for arithmetic it discards, so a run narrow in both
    // takes the scalar kernel.
    const bool reduced = real_axis == 1 && single && lane_kernels_enabled()
        && !any_diffusion(buffers.primal.diffusion, atom_count)
        && !any_diffusion(buffers.diffusion, atom_count);
    const bool atom_lanes =
        reduced && atom_count >= static_cast<std::int64_t>(REAL_LANES);
    const bool lanes = reduced && !atom_lanes
        && train_count >= static_cast<std::int64_t>(REAL_LANES);
    void (*kernel)(
        const JvpBuffers&, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
        std::int64_t
    ) = &simulate_jvp_single_pool<RfMode::INSTANT>;
    if (atom_lanes) {
        kernel = &simulate_real_jvp_atom_lane_range;
    } else if (lanes) {
        kernel = &simulate_real_jvp_lane_range;
    } else if (real_axis == 1 && single) {
        kernel = &simulate_real_jvp_range;
    } else {
        const RfMode mode = buffers.primal.dynamic != nullptr
            ? RfMode::DYNAMIC
            : (buffers.primal.profile != nullptr ? RfMode::PROFILED
                                                     : RfMode::INSTANT);
        if (pools == Pools::TABULATED) {
            kernel = mode == RfMode::DYNAMIC
                ? &simulate_pooled_jvp_range<RfMode::DYNAMIC>
                : (mode == RfMode::PROFILED
                    ? &simulate_pooled_jvp_range<RfMode::PROFILED>
                    : &simulate_pooled_jvp_range<RfMode::INSTANT>);
        } else if (pools == Pools::THREE) {
            kernel = mode == RfMode::DYNAMIC
                ? &simulate_jvp_range<RfMode::DYNAMIC, Pools::THREE>
                : (mode == RfMode::PROFILED
                    ? &simulate_jvp_range<RfMode::PROFILED, Pools::THREE>
                    : &simulate_jvp_range<RfMode::INSTANT, Pools::THREE>);
        } else if (pools == Pools::EXCHANGING) {
            kernel = mode == RfMode::DYNAMIC
                ? &simulate_jvp_range<RfMode::DYNAMIC, Pools::EXCHANGING>
                : (mode == RfMode::PROFILED
                    ? &simulate_jvp_range<RfMode::PROFILED, Pools::EXCHANGING>
                    : &simulate_jvp_range<RfMode::INSTANT, Pools::EXCHANGING>);
        } else if (pools == Pools::SEMISOLID) {
            kernel = mode == RfMode::DYNAMIC
                ? &simulate_jvp_range<RfMode::DYNAMIC, Pools::SEMISOLID>
                : (mode == RfMode::PROFILED
                    ? &simulate_jvp_range<RfMode::PROFILED, Pools::SEMISOLID>
                    : &simulate_jvp_range<RfMode::INSTANT, Pools::SEMISOLID>);
        } else if (mode == RfMode::DYNAMIC) {
            kernel = &simulate_jvp_single_pool<RfMode::DYNAMIC>;
        } else if (mode == RfMode::PROFILED) {
            kernel = &simulate_jvp_single_pool<RfMode::PROFILED>;
        }
    }
    // A lane kernel's work item covers a block of one axis rather than one
    // train: eight atoms of a train, or eight trains of an atom.
    const std::int64_t work_count = atom_lanes
        ? train_count * lane_blocks(atom_count)
        : atom_count * (lanes ? lane_blocks(train_count) : train_count);
    const unsigned int thread_count = worker_count(requested_threads, work_count);
    const std::int64_t block = (work_count + thread_count - 1) / thread_count;
    WorkerPool::instance().run(thread_count, [&](const unsigned int slot) {
        const std::int64_t begin = static_cast<std::int64_t>(slot) * block;
        const std::int64_t end = std::min<std::int64_t>(work_count, begin + block);
        if (begin < end) {
            kernel(buffers, begin, end, event_count, state_count, output_count);
        }
    });
}

PyObject* simulate_jvp(PyObject*, PyObject* arguments) {
    PyObject* pointers = nullptr;
    long long atom_count = 0;
    long long train_count = 0;
    long long event_count = 0;
    long long state_count = 0;
    long long output_count = 0;
    int requested_threads = 0;
    int real_axis = -1;
    double flow_scale = 0.0;
    double washout_scale = 0.0;
    long long shim_count = 1;
    long long locations = 1;
    long long profile_bins = 0;
    double profile_step = 1.0;
    long long lineshape_bins = 0;
    double lineshape_step = 1.0;
    int pool_kind = 0;
    int features = FEATURE_ALL;
    long long pool_count = 0;
    long long pool_rows = 0;
    long long pool_blocks = 0;
    if (!PyArg_ParseTuple(
            arguments,
            "OLLLLLiiddLLLdLdiiLLL",
            &pointers,
            &atom_count,
            &train_count,
            &event_count,
            &state_count,
            &output_count,
            &requested_threads,
            &real_axis,
            &flow_scale,
            &washout_scale,
            &shim_count,
            &locations,
            &profile_bins,
            &profile_step,
            &lineshape_bins,
            &lineshape_step,
            &pool_kind,
            &features,
            &pool_count,
            &pool_rows,
            &pool_blocks
        )) {
        return nullptr;
    }
    const Pools pools = pool_of(pool_kind);
    // The packed buffers, one tangent per differentiable input, the two output
    // planes, then the transition tables and the per-event index that says
    // which an event reads, then the bound pool's lineshape -- all null when
    // the sequence has none.
    constexpr Py_ssize_t expected = PACKED_COUNT + FLOAT_COUNT + 2 + TAIL_COUNT;
    if (!PySequence_Check(pointers) || PySequence_Size(pointers) != expected) {
        PyErr_SetString(PyExc_ValueError, "wrong number of buffer pointers");
        return nullptr;
    }
    if (atom_count < 0 || train_count < 1 || event_count < 0 || state_count < 1
        || output_count < 0) {
        PyErr_SetString(PyExc_ValueError, "invalid EPG buffer dimensions");
        return nullptr;
    }
    void* raw[expected]{};
    for (Py_ssize_t index = 0; index < expected; ++index) {
        if (!parse_pointer(pointers, index, &raw[index])) {
            return nullptr;
        }
    }
    constexpr std::size_t tangents = PACKED_COUNT;
    constexpr std::size_t outputs = tangents + FLOAT_COUNT;
    Buffers primal = packed_buffers(
        raw,
        static_cast<float*>(raw[outputs]),
        static_cast<float*>(raw[outputs + 1]),
        static_cast<std::int64_t>(atom_count),
        static_cast<std::int64_t>(train_count),
        static_cast<float>(flow_scale),
        static_cast<float>(washout_scale),
        static_cast<std::int64_t>(shim_count),
        tail_slot(raw, expected, Tail::TABLE),
        tail_slot(raw, expected, Tail::TABLE_INDEX),
        static_cast<std::int64_t>(profile_bins),
        static_cast<std::int64_t>(locations),
        static_cast<float>(profile_step),
        tail_slot(raw, expected, Tail::PAIRS),
        tail_slot(raw, expected, Tail::PAIR_INDEX),
        tail_slot(raw, expected, Tail::ABSORPTION),
        static_cast<std::int64_t>(lineshape_bins),
        static_cast<float>(lineshape_step),
        features
    );
    if (!attach_pools(
            primal, raw, expected, pool_kind, pool_count, pool_rows, pool_blocks
        )) {
        return nullptr;
    }
    JvpBuffers buffers{};
    buffers.primal = primal;
    // Named rather than positional: an aggregate initializer one entry short
    // leaves the tail null, which the kernel then dereferences. The assert
    // below is what makes a registry change a compile error instead.
    const float** tangent_slots[] = {
        &buffers.t1, &buffers.t2, &buffers.m0, &buffers.b1,
        &buffers.b1_phase, &buffers.b0, &buffers.inversion_efficiency,
        &buffers.diffusion, &buffers.velocity,
        &buffers.bound_fraction, &buffers.exchange_rate, &buffers.t1_bound,
        &buffers.pool_b_fraction, &buffers.pool_b_exchange,
        &buffers.t1_pool_b, &buffers.t2_pool_b, &buffers.pool_b_shift,
        &buffers.duration, &buffers.flip, &buffers.phase,
    };
    static_assert(std::size(tangent_slots) == FLOAT_COUNT, "forward tangents");
    for (std::size_t index = 0; index < FLOAT_COUNT; ++index) {
        *tangent_slots[index] = static_cast<const float*>(raw[tangents + index]);
    }
    buffers.dynamic = static_cast<const float*>(
        tail_slot(raw, expected, Tail::DIRECTION)
    );
    buffers.pool_values = static_cast<const float*>(
        tail_slot(raw, expected, Tail::POOL_DIRECTION)
    );

    Py_BEGIN_ALLOW_THREADS
    dispatch_jvp(
        buffers, atom_count, train_count, event_count, state_count, output_count,
        requested_threads, real_axis, pools
    );
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

PyObject* simulate_vjp(PyObject*, PyObject* arguments) {
    PyObject* pointers = nullptr;
    long long atom_count = 0;
    long long train_count = 0;
    long long event_count = 0;
    long long state_count = 0;
    long long output_count = 0;
    int requested_threads = 0;
    int real_axis = -1;
    double flow_scale = 0.0;
    double washout_scale = 0.0;
    long long shim_count = 1;
    long long locations = 1;
    long long profile_bins = 0;
    double profile_step = 1.0;
    long long lineshape_bins = 0;
    double lineshape_step = 1.0;
    int pool_kind = 0;
    int features = FEATURE_ALL;
    long long pool_count = 0;
    long long pool_rows = 0;
    long long pool_blocks = 0;
    if (!PyArg_ParseTuple(
            arguments,
            "OLLLLLiiddLLLdLdiiLLL",
            &pointers,
            &atom_count,
            &train_count,
            &event_count,
            &state_count,
            &output_count,
            &requested_threads,
            &real_axis,
            &flow_scale,
            &washout_scale,
            &shim_count,
            &locations,
            &profile_bins,
            &profile_step,
            &lineshape_bins,
            &lineshape_step,
            &pool_kind,
            &features,
            &pool_count,
            &pool_rows,
            &pool_blocks
        )) {
        return nullptr;
    }
    // The packed buffers, the two seed planes, one gradient per differentiable
    // input, then the transition tables and the per-event index that says
    // which an event reads -- both null when there is no table -- then the
    // per-voxel rotations, their index, a direction along them and the
    // cotangent that comes back on them, then the bound pool's lineshape.
    constexpr Py_ssize_t expected = PACKED_COUNT + 2 + FLOAT_COUNT + TAIL_COUNT;
    if (!PySequence_Check(pointers) || PySequence_Size(pointers) != expected) {
        PyErr_SetString(PyExc_ValueError, "wrong number of buffer pointers");
        return nullptr;
    }
    if (atom_count < 0 || train_count < 1 || event_count < 0 || state_count < 1
        || output_count < 0) {
        PyErr_SetString(PyExc_ValueError, "invalid EPG buffer dimensions");
        return nullptr;
    }

    void* raw[expected]{};
    for (Py_ssize_t index = 0; index < expected; ++index) {
        if (!parse_pointer(pointers, index, &raw[index])) {
            return nullptr;
        }
    }
    constexpr std::size_t seed = PACKED_COUNT;
    constexpr std::size_t grads = seed + 2;
    Buffers primal = packed_buffers(
        raw, nullptr, nullptr,
        static_cast<std::int64_t>(atom_count),
        static_cast<std::int64_t>(train_count),
        static_cast<float>(flow_scale),
        static_cast<float>(washout_scale),
        static_cast<std::int64_t>(shim_count),
        tail_slot(raw, expected, Tail::TABLE),
        tail_slot(raw, expected, Tail::TABLE_INDEX),
        static_cast<std::int64_t>(profile_bins),
        static_cast<std::int64_t>(locations),
        static_cast<float>(profile_step),
        tail_slot(raw, expected, Tail::PAIRS),
        tail_slot(raw, expected, Tail::PAIR_INDEX),
        tail_slot(raw, expected, Tail::ABSORPTION),
        static_cast<std::int64_t>(lineshape_bins),
        static_cast<float>(lineshape_step),
        features
    );
    if (!attach_pools(
            primal, raw, expected, pool_kind, pool_count, pool_rows, pool_blocks
        )) {
        return nullptr;
    }
    VjpBuffers buffers{};
    buffers.primal = primal;
    buffers.grad_output_real = static_cast<const float*>(raw[seed]);
    buffers.grad_output_imag = static_cast<const float*>(raw[seed + 1]);
    float** const grad_slots[] = {
        &buffers.grad_t1, &buffers.grad_t2, &buffers.grad_m0, &buffers.grad_b1,
        &buffers.grad_b1_phase, &buffers.grad_b0,
        &buffers.grad_inversion_efficiency, &buffers.grad_diffusion,
        &buffers.grad_velocity, &buffers.grad_bound_fraction,
        &buffers.grad_exchange_rate, &buffers.grad_t1_bound,
        &buffers.grad_pool_b_fraction, &buffers.grad_pool_b_exchange,
        &buffers.grad_t1_pool_b, &buffers.grad_t2_pool_b,
        &buffers.grad_pool_b_shift,
        &buffers.grad_flip, &buffers.grad_phase, &buffers.grad_duration,
    };
    static_assert(std::size(grad_slots) == FLOAT_COUNT, "gradient slots");
    for (std::size_t index = 0; index < FLOAT_COUNT; ++index) {
        *grad_slots[index] = static_cast<float*>(raw[grads + index]);
    }
    buffers.grad_dynamic = static_cast<float*>(
        tail_slot(raw, expected, Tail::GRADIENT)
    );
    buffers.grad_pool = static_cast<float*>(
        tail_slot(raw, expected, Tail::POOL_GRADIENT)
    );
    if (tabulated(pool_kind) && buffers.grad_pool == nullptr) {
        PyErr_SetString(PyExc_ValueError, "a tabulated run needs its table's cotangent");
        return nullptr;
    }

    // Lanes over atoms, which is the axis a dictionary is wide in; a run with
    // fewer than a block of them takes the per-atom kernel. Diffusion is left
    // to that one, its damping factors varying with the state as well as the
    // atom.
    const bool atom_lanes = real_axis == 1 && pool_kind == 0
        && lane_kernels_enabled()
        && static_cast<std::int64_t>(atom_count) >= static_cast<std::int64_t>(REAL_LANES)
        && !any_diffusion(buffers.primal.diffusion, atom_count);
    const std::int64_t work_count = atom_lanes
        ? static_cast<std::int64_t>(train_count)
            * lane_blocks(static_cast<std::int64_t>(atom_count))
        : static_cast<std::int64_t>(atom_count)
            * static_cast<std::int64_t>(train_count);
    const unsigned int thread_count = worker_count(requested_threads, work_count);

    const std::size_t events =
        static_cast<std::size_t>(event_count) * static_cast<std::size_t>(train_count);
    const std::size_t atoms = static_cast<std::size_t>(atom_count);
    const TissueLayout layout(buffers.primal.shim_count);
    std::vector<float> shared(3U * events * thread_count, 0.0F);
    std::vector<float> shared_tissue(layout.rows * atoms * thread_count, 0.0F);

    Py_BEGIN_ALLOW_THREADS
    auto slice = [&](unsigned int thread, std::size_t which) {
        return shared.data() + (static_cast<std::size_t>(thread) * 3U + which) * events;
    };
    auto tissue_slice = [&](unsigned int thread) {
        return shared_tissue.data()
            + static_cast<std::size_t>(thread) * layout.rows * atoms;
    };
    {
        void (*kernel)(
            const VjpBuffers&, std::int64_t, std::int64_t, std::int64_t,
            std::int64_t, std::int64_t, float*, float*, float*, float*
        ) = &simulate_vjp_single_pool<RfMode::INSTANT>;
        const RfMode mode = buffers.primal.dynamic != nullptr
            ? RfMode::DYNAMIC
            : (buffers.primal.profile != nullptr ? RfMode::PROFILED
                                                     : RfMode::INSTANT);
        if (tabulated(pool_kind)) {
            kernel = mode == RfMode::DYNAMIC
                ? &simulate_pooled_vjp_range<RfMode::DYNAMIC>
                : (mode == RfMode::PROFILED
                    ? &simulate_pooled_vjp_range<RfMode::PROFILED>
                    : &simulate_pooled_vjp_range<RfMode::INSTANT>);
        } else if (pool_kind == 3) {
            kernel = mode == RfMode::DYNAMIC
                ? &simulate_vjp_range<RfMode::DYNAMIC, Pools::THREE>
                : (mode == RfMode::PROFILED
                    ? &simulate_vjp_range<RfMode::PROFILED, Pools::THREE>
                    : &simulate_vjp_range<RfMode::INSTANT, Pools::THREE>);
        } else if (pool_kind == 2) {
            kernel = mode == RfMode::DYNAMIC
                ? &simulate_vjp_range<RfMode::DYNAMIC, Pools::EXCHANGING>
                : (mode == RfMode::PROFILED
                    ? &simulate_vjp_range<RfMode::PROFILED, Pools::EXCHANGING>
                    : &simulate_vjp_range<RfMode::INSTANT, Pools::EXCHANGING>);
        } else if (pool_kind == 1) {
            kernel = mode == RfMode::DYNAMIC
                ? &simulate_vjp_range<RfMode::DYNAMIC, Pools::SEMISOLID>
                : (mode == RfMode::PROFILED
                    ? &simulate_vjp_range<RfMode::PROFILED, Pools::SEMISOLID>
                    : &simulate_vjp_range<RfMode::INSTANT, Pools::SEMISOLID>);
        } else if (mode == RfMode::DYNAMIC) {
            kernel = &simulate_vjp_single_pool<RfMode::DYNAMIC>;
        } else if (mode == RfMode::PROFILED) {
            kernel = &simulate_vjp_single_pool<RfMode::PROFILED>;
        }
        // The caller establishes the real-subspace conditions and promises not
        // to read the four gradients the representation divides out. A bound
        // pool leaves that subspace, on the same terms as the forward.
        if (atom_lanes) {
            kernel = &simulate_real_vjp_atom_lane_range;
        } else if (real_axis == 1 && pool_kind == 0) {
            kernel = &simulate_real_vjp_range;
        }
        const std::int64_t block = (work_count + thread_count - 1) / thread_count;
        WorkerPool::instance().run(thread_count, [&](const unsigned int slot) {
            const std::int64_t begin = static_cast<std::int64_t>(slot) * block;
            const std::int64_t end = std::min<std::int64_t>(work_count, begin + block);
            if (begin < end) {
                kernel(
                    buffers, begin, end, event_count, state_count, output_count,
                    slice(slot, 0), slice(slot, 1), slice(slot, 2),
                    tissue_slice(slot)
                );
            }
        });
    }
    // Tissue gradients sum over every train, reduced in ascending thread order
    // for the same bitwise reproducibility as the per-event buffers.
    {
        float* const destinations[TISSUE_COUNT] = {
            buffers.grad_t1, buffers.grad_t2, buffers.grad_m0, buffers.grad_b1,
            buffers.grad_b1_phase, buffers.grad_b0,
            buffers.grad_inversion_efficiency, buffers.grad_diffusion,
            buffers.grad_velocity, buffers.grad_bound_fraction,
            buffers.grad_exchange_rate, buffers.grad_t1_bound,
            buffers.grad_pool_b_fraction, buffers.grad_pool_b_exchange,
            buffers.grad_t1_pool_b, buffers.grad_t2_pool_b,
            buffers.grad_pool_b_shift,
        };
        for (std::size_t parameter = 0; parameter < TISSUE_COUNT; ++parameter) {
            const std::size_t rows = tissue_rows(parameter, buffers.primal.shim_count);
            for (std::size_t row = 0; row < rows; ++row) {
                const std::size_t plane = layout.base[parameter] + row;
                for (std::size_t atom = 0; atom < atoms; ++atom) {
                    float total = 0.0F;
                    for (unsigned int thread = 0; thread < thread_count; ++thread) {
                        total += shared_tissue[
                            (static_cast<std::size_t>(thread) * layout.rows + plane)
                                * atoms + atom
                        ];
                    }
                    destinations[parameter][row * atoms + atom] = total;
                }
            }
        }
    }
    // Reduce in ascending thread order so the sum is bitwise reproducible and
    // independent of how the workers were scheduled.
    for (std::size_t event = 0; event < events; ++event) {
        float flip = 0.0F;
        float phase = 0.0F;
        float duration = 0.0F;
        for (unsigned int thread = 0; thread < thread_count; ++thread) {
            flip += shared[(static_cast<std::size_t>(thread) * 3U + 0U) * events + event];
            phase += shared[(static_cast<std::size_t>(thread) * 3U + 1U) * events + event];
            duration += shared[(static_cast<std::size_t>(thread) * 3U + 2U) * events + event];
        }
        buffers.grad_flip[event] = flip;
        buffers.grad_phase[event] = phase;
        buffers.grad_duration[event] = duration;
    }
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

// Spread one forward-over-reverse pass over the pool, then fold the workers'
// partial results into the destination buffers.
void dispatch_second_order(
    const VjpJvpBuffers& buffers,
    const std::int64_t atom_count,
    const std::int64_t train_count,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count,
    const int requested_threads,
    const int real_axis,
    const Pools pools
) {
    const bool bound = pools != Pools::ONE;
    // Lanes run eight trains of one atom at a time, so a run with fewer trains
    // than that fills the rest of the block with repeats of the first and pays
    // for arithmetic it throws away. A dictionary has one train per atom, which
    // is the shape this would cost the most on: the scalar kernel is the faster
    // one there, and measurably so.
    const bool lanes = real_axis == 1 && !bound && lane_kernels_enabled()
        && train_count >= static_cast<std::int64_t>(REAL_LANES)
        && !any_diffusion(buffers.primal.diffusion, atom_count)
        && !any_diffusion(buffers.dot_diffusion, atom_count);
    void (*kernel)(
        const VjpJvpBuffers&, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
        std::int64_t, DualFloat*, DualFloat*, DualFloat*, DualFloat*
    ) = &simulate_vjp_jvp_single_pool<RfMode::INSTANT>;
    if (lanes) {
        kernel = &simulate_real_vjp_jvp_lane_range;
    } else if (real_axis == 1 && !bound) {
        kernel = &simulate_real_vjp_jvp_range;
    } else {
        const RfMode mode = buffers.primal.dynamic != nullptr
            ? RfMode::DYNAMIC
            : (buffers.primal.profile != nullptr ? RfMode::PROFILED
                                                     : RfMode::INSTANT);
        if (pools == Pools::TABULATED) {
            kernel = mode == RfMode::DYNAMIC
                ? &simulate_pooled_vjp_jvp_range<RfMode::DYNAMIC>
                : (mode == RfMode::PROFILED
                    ? &simulate_pooled_vjp_jvp_range<RfMode::PROFILED>
                    : &simulate_pooled_vjp_jvp_range<RfMode::INSTANT>);
        } else if (pools == Pools::THREE) {
            kernel = mode == RfMode::DYNAMIC
                ? &simulate_vjp_jvp_range<RfMode::DYNAMIC, Pools::THREE>
                : (mode == RfMode::PROFILED
                    ? &simulate_vjp_jvp_range<RfMode::PROFILED, Pools::THREE>
                    : &simulate_vjp_jvp_range<RfMode::INSTANT, Pools::THREE>);
        } else if (pools == Pools::EXCHANGING) {
            kernel = mode == RfMode::DYNAMIC
                ? &simulate_vjp_jvp_range<RfMode::DYNAMIC, Pools::EXCHANGING>
                : (mode == RfMode::PROFILED
                    ? &simulate_vjp_jvp_range<RfMode::PROFILED, Pools::EXCHANGING>
                    : &simulate_vjp_jvp_range<RfMode::INSTANT, Pools::EXCHANGING>);
        } else if (pools == Pools::SEMISOLID) {
            kernel = mode == RfMode::DYNAMIC
                ? &simulate_vjp_jvp_range<RfMode::DYNAMIC, Pools::SEMISOLID>
                : (mode == RfMode::PROFILED
                    ? &simulate_vjp_jvp_range<RfMode::PROFILED, Pools::SEMISOLID>
                    : &simulate_vjp_jvp_range<RfMode::INSTANT, Pools::SEMISOLID>);
        } else if (mode == RfMode::DYNAMIC) {
            kernel = &simulate_vjp_jvp_single_pool<RfMode::DYNAMIC>;
        } else if (mode == RfMode::PROFILED) {
            kernel = &simulate_vjp_jvp_single_pool<RfMode::PROFILED>;
        }
    }
    const std::int64_t work_count =
        atom_count * (lanes ? lane_blocks(train_count) : train_count);
    const unsigned int thread_count = worker_count(requested_threads, work_count);

    const std::size_t events =
        static_cast<std::size_t>(event_count) * static_cast<std::size_t>(train_count);
    const std::size_t atoms = static_cast<std::size_t>(atom_count);
    // Workers take disjoint trains, and an event gradient belongs to exactly one
    // train, so all of them accumulate into one buffer without sharing a slot.
    // Only the tissue gradients, which every train contributes to, need a copy
    // per worker and a reduction.
    const TissueLayout layout(buffers.primal.shim_count);
    std::vector<DualFloat> shared(3U * events, DualFloat{0.0F, 0.0F});
    std::vector<DualFloat> shared_tissue(
        layout.rows * atoms * thread_count, DualFloat{0.0F, 0.0F}
    );
    auto slice = [&](std::size_t which) { return shared.data() + which * events; };
    auto tissue_slice = [&](unsigned int slot) {
        return shared_tissue.data()
            + static_cast<std::size_t>(slot) * layout.rows * atoms;
    };
    {
        // Round the block up to a whole number of trains so no two workers hold
        // the same train's event gradients.
        const std::int64_t trains_per_block =
            (work_count / atom_count + thread_count - 1) / thread_count;
        const std::int64_t block = trains_per_block * atom_count;
        WorkerPool::instance().run(thread_count, [&](const unsigned int slot) {
            const std::int64_t begin = static_cast<std::int64_t>(slot) * block;
            const std::int64_t end = std::min<std::int64_t>(work_count, begin + block);
            if (begin < end) {
                kernel(
                    buffers, begin, end, event_count, state_count, output_count,
                    slice(0), slice(1), slice(2), tissue_slice(slot)
                );
            }
        });
    }
    {
        float* const value_destinations[TISSUE_COUNT] = {
            buffers.grad_dot_t1, buffers.grad_dot_t2, buffers.grad_dot_m0,
            buffers.grad_dot_b1, buffers.grad_dot_b1_phase, buffers.grad_dot_b0,
            buffers.grad_dot_inversion_efficiency, buffers.grad_dot_diffusion,
            buffers.grad_dot_velocity, buffers.grad_dot_bound_fraction,
            buffers.grad_dot_exchange_rate, buffers.grad_dot_t1_bound,
            buffers.grad_dot_pool_b_fraction, buffers.grad_dot_pool_b_exchange,
            buffers.grad_dot_t1_pool_b, buffers.grad_dot_t2_pool_b,
            buffers.grad_dot_pool_b_shift,
        };
        float* const tangent_destinations[TISSUE_COUNT] = {
            buffers.grad_t1, buffers.grad_t2, buffers.grad_m0, buffers.grad_b1,
            buffers.grad_b1_phase, buffers.grad_b0,
            buffers.grad_inversion_efficiency, buffers.grad_diffusion,
            buffers.grad_velocity, buffers.grad_bound_fraction,
            buffers.grad_exchange_rate, buffers.grad_t1_bound,
            buffers.grad_pool_b_fraction, buffers.grad_pool_b_exchange,
            buffers.grad_t1_pool_b, buffers.grad_t2_pool_b,
            buffers.grad_pool_b_shift,
        };
        for (std::size_t parameter = 0; parameter < TISSUE_COUNT; ++parameter) {
            const std::size_t rows = tissue_rows(parameter, buffers.primal.shim_count);
            for (std::size_t row = 0; row < rows; ++row) {
                const std::size_t plane = layout.base[parameter] + row;
                for (std::size_t atom = 0; atom < atoms; ++atom) {
                    DualFloat total{0.0F, 0.0F};
                    for (unsigned int slot = 0; slot < thread_count; ++slot) {
                        total = total + shared_tissue[
                            (static_cast<std::size_t>(slot) * layout.rows + plane)
                                * atoms + atom
                        ];
                    }
                    value_destinations[parameter][row * atoms + atom] = total.value;
                    tangent_destinations[parameter][row * atoms + atom] = total.tangent;
                }
            }
        }
    }
    for (std::size_t event = 0; event < events; ++event) {
        const DualFloat flip = shared[event];
        const DualFloat phase = shared[events + event];
        const DualFloat duration = shared[2U * events + event];
        buffers.grad_dot_flip[event] = flip.value;
        buffers.grad_flip[event] = flip.tangent;
        buffers.grad_dot_phase[event] = phase.value;
        buffers.grad_phase[event] = phase.tangent;
        buffers.grad_dot_duration[event] = duration.value;
        buffers.grad_duration[event] = duration.tangent;
    }
}

PyObject* simulate_vjp_jvp(PyObject*, PyObject* arguments) {
    PyObject* pointers = nullptr;
    long long atom_count = 0;
    long long train_count = 0;
    long long event_count = 0;
    long long state_count = 0;
    long long output_count = 0;
    int requested_threads = 0;
    int real_axis = -1;
    double flow_scale = 0.0;
    double washout_scale = 0.0;
    long long shim_count = 1;
    long long locations = 1;
    long long profile_bins = 0;
    double profile_step = 1.0;
    long long lineshape_bins = 0;
    double lineshape_step = 1.0;
    int pool_kind = 0;
    int features = FEATURE_ALL;
    long long pool_count = 0;
    long long pool_rows = 0;
    long long pool_blocks = 0;
    if (!PyArg_ParseTuple(
            arguments,
            "OLLLLLiiddLLLdLdiiLLL",
            &pointers,
            &atom_count,
            &train_count,
            &event_count,
            &state_count,
            &output_count,
            &requested_threads,
            &real_axis,
            &flow_scale,
            &washout_scale,
            &shim_count,
            &locations,
            &profile_bins,
            &profile_step,
            &lineshape_bins,
            &lineshape_step,
            &pool_kind,
            &features,
            &pool_count,
            &pool_rows,
            &pool_blocks
        )) {
        return nullptr;
    }
    // The packed buffers, the tangents, the two seed planes, both gradient
    // blocks, then the transition tables and the per-event index that says
    // which an event reads -- both null when there is no table -- and the
    // bound pool's lineshape, null when there is no bound pool.
    constexpr Py_ssize_t expected = PACKED_COUNT + 3 * FLOAT_COUNT + 2 + TAIL_COUNT;
    if (!PySequence_Check(pointers) || PySequence_Size(pointers) != expected) {
        PyErr_SetString(PyExc_ValueError, "wrong number of buffer pointers");
        return nullptr;
    }
    if (atom_count < 0 || train_count < 1 || event_count < 0 || state_count < 1
        || output_count < 0) {
        PyErr_SetString(PyExc_ValueError, "invalid EPG buffer dimensions");
        return nullptr;
    }
    const Pools pools = pool_of(pool_kind);

    void* raw[expected]{};
    for (Py_ssize_t index = 0; index < expected; ++index) {
        if (!parse_pointer(pointers, index, &raw[index])) {
            return nullptr;
        }
    }
    Buffers primal = packed_buffers(
        raw, nullptr, nullptr,
        static_cast<std::int64_t>(atom_count),
        static_cast<std::int64_t>(train_count),
        static_cast<float>(flow_scale),
        static_cast<float>(washout_scale),
        static_cast<std::int64_t>(shim_count),
        tail_slot(raw, expected, Tail::TABLE),
        tail_slot(raw, expected, Tail::TABLE_INDEX),
        static_cast<std::int64_t>(profile_bins),
        static_cast<std::int64_t>(locations),
        static_cast<float>(profile_step),
        tail_slot(raw, expected, Tail::PAIRS),
        tail_slot(raw, expected, Tail::PAIR_INDEX),
        tail_slot(raw, expected, Tail::ABSORPTION),
        static_cast<std::int64_t>(lineshape_bins),
        static_cast<float>(lineshape_step),
        features
    );
    if (!attach_pools(
            primal, raw, expected, pool_kind, pool_count, pool_rows, pool_blocks
        )) {
        return nullptr;
    }
    VjpJvpBuffers buffers{};
    buffers.primal = primal;
    constexpr std::size_t tangents = PACKED_COUNT;
    constexpr std::size_t seed = tangents + FLOAT_COUNT;
    constexpr std::size_t values = seed + 2;
    constexpr std::size_t tangent_grads = values + FLOAT_COUNT;
    const float** tangent_slots[] = {
        &buffers.dot_t1, &buffers.dot_t2, &buffers.dot_m0, &buffers.dot_b1,
        &buffers.dot_b1_phase, &buffers.dot_b0,
        &buffers.dot_inversion_efficiency, &buffers.dot_diffusion,
        &buffers.dot_velocity, &buffers.dot_bound_fraction,
        &buffers.dot_exchange_rate, &buffers.dot_t1_bound,
        &buffers.dot_pool_b_fraction, &buffers.dot_pool_b_exchange,
        &buffers.dot_t1_pool_b, &buffers.dot_t2_pool_b,
        &buffers.dot_pool_b_shift,
        &buffers.dot_duration, &buffers.dot_flip, &buffers.dot_phase,
    };
    static_assert(std::size(tangent_slots) == FLOAT_COUNT, "tangent slots");
    for (std::size_t index = 0; index < FLOAT_COUNT; ++index) {
        *tangent_slots[index] = static_cast<const float*>(raw[tangents + index]);
    }
    buffers.grad_output_real = static_cast<const float*>(raw[seed]);
    buffers.grad_output_imag = static_cast<const float*>(raw[seed + 1]);
    float** value_slots[] = {
        &buffers.grad_dot_t1, &buffers.grad_dot_t2, &buffers.grad_dot_m0,
        &buffers.grad_dot_b1, &buffers.grad_dot_b1_phase, &buffers.grad_dot_b0,
        &buffers.grad_dot_inversion_efficiency, &buffers.grad_dot_diffusion,
        &buffers.grad_dot_velocity, &buffers.grad_dot_bound_fraction,
        &buffers.grad_dot_exchange_rate, &buffers.grad_dot_t1_bound,
        &buffers.grad_dot_pool_b_fraction, &buffers.grad_dot_pool_b_exchange,
        &buffers.grad_dot_t1_pool_b, &buffers.grad_dot_t2_pool_b,
        &buffers.grad_dot_pool_b_shift,
        &buffers.grad_dot_duration, &buffers.grad_dot_flip,
        &buffers.grad_dot_phase,
    };
    float** tangent_grad_slots[] = {
        &buffers.grad_t1, &buffers.grad_t2, &buffers.grad_m0, &buffers.grad_b1,
        &buffers.grad_b1_phase, &buffers.grad_b0,
        &buffers.grad_inversion_efficiency, &buffers.grad_diffusion,
        &buffers.grad_velocity, &buffers.grad_bound_fraction,
        &buffers.grad_exchange_rate, &buffers.grad_t1_bound,
        &buffers.grad_pool_b_fraction, &buffers.grad_pool_b_exchange,
        &buffers.grad_t1_pool_b, &buffers.grad_t2_pool_b,
        &buffers.grad_pool_b_shift,
        &buffers.grad_duration, &buffers.grad_flip, &buffers.grad_phase,
    };
    static_assert(std::size(value_slots) == FLOAT_COUNT, "value slots");
    static_assert(
        std::size(tangent_grad_slots) == FLOAT_COUNT, "tangent gradient slots"
    );
    for (std::size_t index = 0; index < FLOAT_COUNT; ++index) {
        *value_slots[index] = static_cast<float*>(raw[values + index]);
        *tangent_grad_slots[index] =
            static_cast<float*>(raw[tangent_grads + index]);
    }
    buffers.dynamic = static_cast<const float*>(
        tail_slot(raw, expected, Tail::DIRECTION)
    );
    buffers.grad_dot_dynamic = static_cast<float*>(
        tail_slot(raw, expected, Tail::GRADIENT)
    );
    buffers.grad_dynamic = static_cast<float*>(
        tail_slot(raw, expected, Tail::CURVATURE)
    );
    buffers.pool_values = static_cast<const float*>(
        tail_slot(raw, expected, Tail::POOL_DIRECTION)
    );
    buffers.grad_dot_pool = static_cast<float*>(
        tail_slot(raw, expected, Tail::POOL_GRADIENT)
    );
    buffers.grad_pool = static_cast<float*>(
        tail_slot(raw, expected, Tail::POOL_CURVATURE)
    );
    if (tabulated(pool_kind)
        && (buffers.grad_dot_pool == nullptr || buffers.grad_pool == nullptr)) {
        PyErr_SetString(PyExc_ValueError, "a tabulated run needs its table's cotangent");
        return nullptr;
    }

    Py_BEGIN_ALLOW_THREADS
    dispatch_second_order(
        buffers, atom_count, train_count, event_count, state_count, output_count,
        requested_threads, real_axis, pools
    );
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}


PyMethodDef methods[] = {
    {"simulate", simulate, METH_VARARGS, "Run a fused CPU EPG state machine."},
    {"simulate_jvp", simulate_jvp, METH_VARARGS, "Run a fused CPU EPG JVP."},
    {"simulate_vjp", simulate_vjp, METH_VARARGS, "Run a fused CPU EPG VJP."},
    {"simulate_vjp_jvp", simulate_vjp_jvp, METH_VARARGS,
     "Run a fused CPU EPG forward-over-reverse pass."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_epg_cpu",
    "Torch-free CPU EPG kernels.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__epg_cpu() {
    PyObject* const created = PyModule_Create(&module);
    if (created == nullptr) {
        return nullptr;
    }
    // What a caller cannot see from the outside: whether this build has
    // several instruction sets to pick a kernel from, and so whether the
    // vector state a pool job leaves behind is a thing that exists here.
    if (PyModule_AddIntConstant(created, "multiversioned", TORCHSIM_MULTIVERSIONED)
        < 0) {
        Py_DECREF(created);
        return nullptr;
    }
    return created;
}
