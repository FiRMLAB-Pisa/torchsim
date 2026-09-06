"""What the spins are, and what each kind of event does to them.

A signal model made of a state machine splits in two. A
:class:`SpinPhysics` says what a voxel holds -- which tissue properties
are exposed, and so which physics the kernels carry -- and what each kind of
event is realized as: whether an excitation is ideal or integrated from a
waveform, whether a readout is followed by an unbalanced gradient, by ideal
spoiling, or by nothing. A :class:`Simulator` says what order the
events are played in.

Splitting them is what lets one be changed without the other. The MRF timing
with a selective excitation, or a refocused train whose readout spoils rather
than winds, is an assignment rather than a new model.

**The operators are resolved before a description exists.** A simulator binds
each slot when it is constructed; what a protocol then produces is an ordinary
:class:`~torchsim.sequence.SequenceDescription`, whose events carry their own
action word. From there the path is the fused one -- packing, the feature
mask, the real-subspace verdict, offload and sharding -- and nothing consults
an operator slot again. There is no interpretation at run time and none per
event.

Three vocabularies name a pulse along that path, and they are not the same
vocabulary at three sizes. :class:`EventOperators` has one slot per role a
sequence is written in terms of, and its values are operator factories, read
only while a description is being assembled. Each factory emits events tagged
with an :class:`~torchsim.sequence.RfUse`, which is what a Pulseq file
carries, and with an :class:`~torchsim.sequence.EventAction`, which is the bit
field the kernels read. The three do not line up one to one and are not meant
to: the :attr:`~EventOperators.saturation` slot plays a pulse tagged
``RfUse.EXCITATION``, because what the scanner is told about a pulse and what
role the sequence gives it are separate questions.
"""

from __future__ import annotations

__all__ = [
    "Simulator",
    "BALANCED",
    "REFOCUSED",
    "SPOILED",
    "SpinPhysics",
    "EventOperators",
    "UNBALANCED",
]

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from copy import copy as shallow_copy
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

import torch

from ..sequence import (
    Delay,
    EpgEngine,
    EventType,
    Excitation,
    FSEReadout,
    Inversion,
    Operator,
    Readout,
    Refocusing,
    RfDefinition,
    RfUse,
    Saturation,
    SequenceDescription,
    SequenceEvent,
    ShimDefinition,
    SPGRReadout,
    SSFPFidReadout,
    TissueProperties,
    bSSFPReadout,
    compose,
    execution,
    ideal_rf_definition,
)
from ..sequence._array import brought, is_array, read
from ..sequence._parameters import PROPERTY_NAMES, PUBLIC_PROPERTIES
from ..sequence._simulation import RecordMode, target_device
from ..sequence._transition import across_the_slice
from ._binding import Packing, bind, run_key
from ._signal import _moved, _SignalModel

_EMPTY: Mapping[str, Any] = MappingProxyType({})

# Stands for a setting the caller did not name, so that the class body is
# what decides -- None being a value several of them take.
_UNSET: Any = object()

# What a caller may name that describes the run rather than the sequence, at
# the constructor, on bind() or at the call. Each is spelled the same in all
# three, and "device" is the one a run takes without holding.
RUN_SETTINGS = (
    "states",
    "repetitions",
    "record",
    "device",
    "execution",
    "pulse",
    "shims",
    "across_slice",
)

# The raster :class:`~torchsim.sequence.EpgEngine` reads a pulse's shape on,
# named here because a packing resolved against one is not valid against
# another.
_RF_RASTER_TIME_S = 1e-6


def realised(
    description: SequenceDescription, physics: SpinPhysics
) -> SequenceDescription:
    """An arriving event stream, re-emitted through a model's own handlers.

    The transport carries RF pulses and ADC windows and no gradients, so a
    description that arrives says what was played and not how the sequence
    dephased between one event and the next. That belongs to the sequence
    family rather than to the stream, and it is what the operators hold: a
    refocusing pulse brings its crusher pair, an unbalanced sample winds an
    order after it, a spoiled one discards the transverse states.

    So each event is played back through the operator its kind and its RF use
    name, at the timestamp it arrived with. A gradient hangs off a pulse or a
    sample, which is what the wire carries; one standing on its own -- the
    spoiler of a preparation written by hand -- has no handler to reinstate it
    and does not survive, because an arriving stream could not have held it
    either. A pulse whose use is one the
    handlers have no reading for -- a preparation, or an untagged one -- is
    emitted as it stands, with no gradient behaviour added, since guessing one
    is how a stream comes back as the wrong sequence.
    """
    parts: list[tuple[Any, Any]] = []
    for event in description.events:
        when = float(event.timestamp_us) * 1e-6
        if event.type is EventType.RF:
            parts.append((when, _rf_operator(event, physics.operators)))
        elif event.type is EventType.ADC:
            parts.append((when, _adc_operator(event, physics.operators)))
    if not parts:
        return description
    events, _played_s = compose(*parts)
    return replace(description, events=events)


#: The engine's spelling for a setting this layer takes under another name.
_RENAMED = MappingProxyType({"nstates": "states"})


def _no_renamed(values: Mapping[str, Any]) -> None:
    """Refuse a run setting under the name the engine takes it by.

    Held quietly instead, it would become a protocol argument and reach a
    layout that has no parameter for it -- or a closed form that ignores it,
    and answers with the wrong number of orders.
    """
    for given in _RENAMED.keys() & values.keys():
        raise TypeError(
            f"{given!r} is EpgEngine's name for this setting; a simulator "
            f"takes it as {_RENAMED[given]!r}, on the constructor, on bind() "
            f"or at the call"
        )


def _named(given: Any, declared: Any) -> Any:
    """The value the caller named, or the one the class body declares."""
    return declared if given is _UNSET else given


def _accepted(handler: Any, **offered: Any) -> dict[str, Any]:
    """The offered arguments this handler has somewhere to put.

    The handlers differ in what a stream can tell them: a readout that fixes
    the role it records at takes none, an inversion is a pulse whose flip is
    its own. Passing what a handler does not take is how a stream that carries
    more than one family of sequence stops working on the second one.
    """
    from inspect import signature

    takes = signature(handler).parameters
    return {name: value for name, value in offered.items() if name in takes}


def _adc_operator(event: SequenceEvent, operators: EventOperators) -> Any:
    """The sample this model's readout makes of an ADC window."""
    return operators.readout(
        event.adc_phase_rad,
        **_accepted(operators.readout, role=event.adc_role, is_echo=event.is_echo),
    )


def _rf_operator(event: SequenceEvent, operators: EventOperators) -> Any:
    """The operator a pulse's own ``use`` tag names."""
    handler = {
        RfUse.REFOCUSING: operators.refocusing,
        RfUse.INVERSION: operators.inversion,
        RfUse.SATURATION: operators.saturation,
    }.get(event.rf_use, operators.excitation)
    offered = _accepted(
        handler,
        flip_rad=event.rf_amplitude_hz,
        phase_rad=event.rf_phase_rad,
        definition_id=event.rf_definition_id,
        frequency_hz=event.rf_frequency_hz,
        offset_hz=event.rf_frequency_hz,
        shim_id=event.rf_shim_id,
    )
    return handler(**offered)


def replace_pulse(simulator: Simulator, pulse: RfDefinition) -> Simulator:
    """A copy of ``simulator`` whose events drive ``pulse``.

    The shipped operators name definition zero, so substituting there is what
    makes a shaped pulse the one they play.
    """
    held = shallow_copy(simulator)
    held.model = replace(simulator.model, definitions={0: replace(pulse, id=0)})
    held._packing = None
    held._described = None
    return held


#: The kinds of event a model says what to do with.
_HANDLER_SLOTS = (
    "excitation",
    "refocusing",
    "inversion",
    "saturation",
    "readout",
    "delay",
)


@dataclass(frozen=True)
class EventOperators:
    """Which operator plays each kind of event.

    Each field is an operator factory, called with the parameters the protocol
    has for that event. Assigning one is how a sequence says that its readouts
    wind the states on, or that its excitation is a shaped pulse rather than an
    ideal rotation.

    These are roles a sequence is written in terms of, not the tags the events
    end up carrying: a factory here decides which
    :class:`~torchsim.sequence.RfUse` and which
    :class:`~torchsim.sequence.EventAction` its events are emitted with.
    """

    #: What a pulse that tips magnetization into the transverse plane plays.
    excitation: Callable[..., Operator] = Excitation
    #: What a refocusing pulse plays, including the gradients it sits between.
    refocusing: Callable[..., Operator] = Refocusing
    #: What an inversion plays, and the recovery it holds the timeline for.
    inversion: Callable[..., Operator] = Inversion
    #: What a saturation pulse plays.
    saturation: Callable[..., Operator] = Saturation
    #: What a sample and the rest of its repetition play -- which is where a
    #: sequence says whether it winds the states on, spoils them, or rewinds
    #: them.
    readout: Callable[..., Operator] = Readout
    #: What a wait plays, which is nothing but time.
    delay: Callable[..., Operator] = Delay


def _uncrushed(*args: Any, **kwargs: Any) -> Operator:
    """Return a refocusing pulse with no crushers, as a balanced sequence plays."""
    return Refocusing(*args, crushed=False, **kwargs)


#: A readout the repetition rewinds after, and refocusing pulses left uncrushed.
BALANCED = EventOperators(readout=bSSFPReadout, refocusing=_uncrushed)
#: A readout followed by one unbalanced gradient.
UNBALANCED = EventOperators(readout=SSFPFidReadout)
#: A readout followed by ideal transverse spoiling.
SPOILED = EventOperators(readout=SPGRReadout)
#: A refocusing pulse between its crushers, and the sample at the echo centre.
REFOCUSED = EventOperators(readout=FSEReadout)


@dataclass(frozen=True)
class SpinPhysics:
    """Which properties a voxel has, and what each kind of event does to it.

    Attributes
    ----------
    properties : mapping
        ``{public name: tissue field}``. A field left unnamed is never given to
        the tissue, and the kernels leave its term out -- so this is how a
        model asks for off-resonance, diffusion, flow or a second pool. A name
        mapped to ``None`` is the model's own and reaches the signal without
        reaching the tissue.
    operators : EventOperators
        What each kind of event plays.
    fixed : mapping
        Tissue fields the model pins rather than exposes, as
        ``{field: value}``.
    definitions : mapping
        The RF resources the events name. The default is one ideal hard pulse
        at id 0; a model whose excitation is slice-selective supplies a shaped
        definition here instead.
    """

    properties: Mapping[str, str | None] = field(default_factory=lambda: _EMPTY)
    operators: EventOperators = field(default_factory=EventOperators)
    fixed: Mapping[str, Any] = field(default_factory=lambda: _EMPTY)
    definitions: Mapping[int, RfDefinition] = field(
        default_factory=lambda: {0: ideal_rf_definition()}
    )

    def tissue(self, properties: Mapping[str, Any]) -> TissueProperties:
        """Build the tissue, leaving everything undeclared at its identity.

        A property the model does not expose, and one it exposes that the
        caller left out, are both simply absent -- so each reaches the gate as
        the scalar default :class:`TissueProperties` holds and the kernels
        leave its term out.

        Raises
        ------
        ValueError
            If a name is mapped to a field the tissue does not have.
        """
        values = dict(self.fixed)
        values.update(
            {
                self.fields[name]: value
                for name, value in properties.items()
                if self.fields.get(name) is not None
            }
        )
        return TissueProperties(**values)

    @property
    def fields(self) -> dict[str, str | None]:
        """The public-name to tissue-field map, checked against the tissue.

        Every field a voxel has can be named, whether or not this model asked
        for it: the vocabulary is the same for all of them, and giving a value
        is what turns a term on. What ``properties`` adds is a model's own
        spelling -- a name of its own for a field, or a name it answers to and
        the tissue does not.
        """
        pairs = {**PUBLIC_PROPERTIES, **self.properties}
        unknown = {field for field in pairs.values() if field is not None}
        unknown |= set(self.fixed)
        unknown -= set(PROPERTY_NAMES)
        if unknown:
            raise ValueError(f"unknown tissue: {sorted(unknown)}")
        return pairs


class Simulator(_SignalModel):
    """A protocol: what a sequence plays, and the physics behind it.

    The one thing anything downstream takes. Parameter inference, model-based
    reconstruction and sequence design are written against this and never ask
    how the signal is arrived at.

    Subclasses set :attr:`model` and implement :meth:`layout`, which returns
    the operators of one repetition in the order they are played. A protocol
    with a closed form -- a steady state that needs no state machine --
    implements :meth:`evaluate` instead and never reaches :meth:`layout`; its
    :attr:`model` then carries only the property declaration, since there are
    no events for operators to realize.

    **The constructor takes the keywords** :meth:`simulate` **takes, and fixes
    them; a call overrides.** So a sequence is written once with the tissue it
    is being asked about already on it, and what is left to give per call is
    whatever is actually varying -- the design under optimization, the map
    being fitted.

    A model that composes others rather than declaring physics of its own
    names its properties in the class body and writes whatever constructor
    suits it: every setting has a value at the class level, so one that never
    reaches this constructor still answers :meth:`simulate`.

    Attributes
    ----------
    model : SpinPhysics, optional
        The physics behind the protocol.
    states : int, optional
        Configuration orders to carry, or ``None`` to size them from the
        winding the description asks for.
    """

    model: SpinPhysics = SpinPhysics()
    states: int | None = None
    # How many playings a sequence needs to reach the state a scanner plays it
    # in. One is the transient from equilibrium, which is what a scanner plays
    # once and never again; a sequence whose own physics says otherwise
    # overrides this.
    repetitions: int = 1
    record: RecordMode = "all"
    execution: str | torch.device | Sequence[Any] | None = None
    across_slice: Any = None
    crusher_dephasing_rad: float = 0.0
    voxel_size_m: float | None = None

    # Every attribute the constructor sets has a value here as well, so a
    # closed form that writes an __init__ of its own -- taking the two blocks
    # it concatenates, say -- still answers simulate() without chaining to
    # this one. What such a subclass declares in its class body is then the
    # whole of what it is.
    protocol: Mapping[str, Any] = _EMPTY
    shims: Mapping[int, ShimDefinition] = MappingProxyType({})
    _brought: Any = None
    _described: SequenceDescription | None = None
    _packing: Packing | None = None
    _refused: Sequence[Any] = ()
    _resolving: bool = True

    def __new__(cls, *args: Any, **kwargs: Any) -> Simulator:
        """Refuse the base class itself.

        Nothing is named here: no physics, no handlers, and neither a layout
        nor a closed form. A sequence is what a subclass says.
        """
        if cls is Simulator:
            raise TypeError(
                "Simulator is what a sequence is written against, not a "
                "sequence: subclass it and implement layout() for a train of "
                "events, or evaluate() for a closed form"
            )
        return super().__new__(cls)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Read handlers named in the class body into the model.

        A subclass may say what plays each kind of event by naming it, which
        is the shortest way to write a sequence family:

        .. code-block:: python

            class SSFPMRF(Simulator):
                excitation = Excitation
                readout = SSFPFidReadout

        Anything not named keeps what the base class had, and a ``model``
        given outright still wins, which is what a sequence needs when it also
        fixes tissue or carries a pulse shape.
        """
        super().__init_subclass__(**kwargs)
        named = {
            slot: cls.__dict__[slot] for slot in _HANDLER_SLOTS if slot in cls.__dict__
        }
        if named and "model" not in cls.__dict__:
            cls.model = replace(
                cls.model, operators=replace(cls.model.operators, **named)
            )
            for slot in named:
                delattr(cls, slot)
        # The physics names what the model exposes, unless the class body did
        # -- which is what a model composing others, with no physics of its
        # own, does.
        if "properties" not in cls.__dict__:
            cls.properties = cls.model.properties

    def __init__(
        self,
        *,
        model: SpinPhysics | None = None,
        states: int | None = None,
        repetitions: int | str | None = None,
        record: RecordMode = _UNSET,
        execution: str | torch.device | Sequence[Any] | None = _UNSET,
        pulse: RfDefinition | None = None,
        shims: Mapping[int, ShimDefinition] | None = None,
        across_slice: Any = _UNSET,
        resolve: bool = True,
        crusher_dephasing_rad: float = _UNSET,
        voxel_size_m: float | None = _UNSET,
        **protocol: Any,
    ) -> None:
        """Bind the physics, the protocol and any tissue this simulator plays.

        Parameters
        ----------
        model:
            The physics, or ``None`` for the class's own.
        states:
            Configuration orders to carry.
        repetitions:
            How many times the description is played to reach the state a
            scanner plays it in, of which the last is the one recorded. One --
            the default, unless the sequence declares otherwise -- records the
            playing that starts from equilibrium, which is the transient a
            scanner plays once and never again. ``"auto"`` reads the settled
            state off a handful of playings rather than running to it, and
            holds no structure fixed across calls.
        record:
            Which ADCs the signal holds.
        execution:
            Where to run -- ``"auto"`` to decide per call against what the
            devices have free, ``"cpu"``, or a device or list of devices.
            ``None`` follows whatever :func:`~torchsim.sequence.execution`
            block is in scope, which is what lets a caller decide instead.
        pulse:
            The waveform the events drive, taking the place of the ideal hard
            rotation the shipped operators name. Giving one is what makes the
            layout's pulses shaped; where across the slice to work them out is
            ``across_slice``.
        shims:
            The transmit shims the pulses are driven on, by id, for a layout
            whose operators name a ``shim_id``. One channel driven alike when
            not given.
        across_slice:
            How many positions across the slice to integrate a shaped pulse
            at, or an ``exact_slice_profile()`` saying which. ``None`` works
            it out at the slice centre alone, which is the hard-pulse answer.
        resolve:
            Whether to hold the protocol's structure fixed across calls, so
            that a call which changes only numbers rebinds them onto events
            already packed. A loop that plays the same sequence with
            different numbers -- a design, a dictionary sweep -- is worth
            roughly eight times the whole call this way. Turning it off
            rebuilds the event stream every call, which is slower and agrees
            to the last bit rather than to float32 round-off.
        crusher_dephasing_rad, voxel_size_m:
            The unbalanced gradient the sequence plays, and the voxel it winds
            across. Their ratio is what diffusion is damped by and what flow
            turns each dephasing order through.
        protocol:
            The sequence arguments :meth:`layout` reads, and any tissue
            property to fix, under the names :attr:`properties` declares.
        """
        _no_renamed(protocol)
        self.model = model if model is not None else type(self).model
        if pulse is not None:
            self.model = replace(self.model, definitions={0: replace(pulse, id=0)})
        if model is not None:
            # A physics given here names the properties; otherwise the class
            # body already said them, outright or through its own physics.
            self.properties = self.model.properties
        self.states = states if states is not None else type(self).states
        self.repetitions = (
            repetitions if repetitions is not None else type(self).repetitions
        )
        # Each falls back to what the class body says, which is where a
        # sequence that always records one shot or always spans one slice
        # states it once rather than in a constructor of its own.
        self.record = _named(record, type(self).record)
        self.execution = _named(execution, type(self).execution)
        self.crusher_dephasing_rad = _named(
            crusher_dephasing_rad, type(self).crusher_dephasing_rad
        )
        self.voxel_size_m = _named(voxel_size_m, type(self).voxel_size_m)
        self.shims = dict(shims) if shims else {}
        self.across_slice = across_the_slice(
            _named(across_slice, type(self).across_slice)
        )
        self._resolving = bool(resolve)
        self._packing = None
        self._refused = ()
        # Read once, so a layout can be written in torch whatever the caller
        # brought, and so the answer knows where to go back to.
        self._brought = brought(protocol.values())
        # Split the way a call is split, so the constructor takes exactly what
        # simulate() takes and fixes it.
        declared = set(self.accepts)
        self.bound = self._fix(
            {name: value for name, value in protocol.items() if name in declared}
        )
        self.protocol = read(
            {
                name: value
                for name, value in protocol.items()
                if name not in declared and name not in RUN_SETTINGS
            }
        )
        self._described = None

    def bind(self, **values: Any) -> Simulator:
        """This simulator with more fixed on it, values or settings alike.

        A property or a protocol argument is held for the next call; a setting
        -- the pulse the events drive, where across the slice to work it out,
        how many orders to carry -- is applied to the copy instead, because it
        changes what is simulated rather than what is simulated with.
        """
        _no_renamed(values)
        settings = {name: values.pop(name) for name in RUN_SETTINGS if name in values}
        held = super().bind(**values)
        pulse = settings.pop("pulse", None)
        if pulse is not None:
            held = replace_pulse(held, pulse)
        if "across_slice" in settings:
            held.across_slice = across_the_slice(settings.pop("across_slice"))
        for name, value in settings.items():
            setattr(held, name, value)
        return held

    @property
    def operators(self) -> EventOperators:
        """What plays each kind of event, from this simulator's physics.

        A layout reads its operators here. Each is resolved while a
        description is being assembled and never again: what the layout
        produces carries its own action word, and the run consults no slot.
        """
        return self.model.operators

    @property
    def variables(self) -> tuple[str, ...]:
        """The protocol arguments this simulator's layout takes.

        What a sequence is written in, as against the tissue it is played on:
        :attr:`exposes` and :attr:`accepts` name the properties, this names the
        flip angles, spacings and times. Everything here can be fixed on the
        constructor, given at the call, or carried as a tensor a cost is
        differentiated back through.
        """
        from inspect import Parameter, signature

        return tuple(
            name
            for name, parameter in signature(self.layout).parameters.items()
            if parameter.kind not in (Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL)
        )

    def to(self, device: torch.device | str) -> Simulator:
        """This simulator, with everything it holds on ``device``.

        A simulator carries its protocol -- echo times, a flip train -- and
        whatever tissue is fixed on it, and the two have to arrive on a card
        together: properties moved on their own would be multiplied against
        echo times still on the host.

        Parameters
        ----------
        device : torch.device or str
            Where to put it.

        Returns
        -------
        Simulator
            A copy. This one is left where it was.
        """
        moved = super().to(device)
        moved.protocol = _moved(moved.protocol, torch.device(device))
        # Whatever was resolved was resolved somewhere else, and holds tensors
        # that live there.
        moved._packing = None
        moved._refused = ()
        return moved

    def _structure(
        self,
        played: Mapping[str, Any],
        tissue: TissueProperties,
        *,
        repetitions: int | str,
        record: str,
        device: Any,
        across_slice: Any = None,
    ) -> tuple[SequenceDescription, Any]:
        """The description to run, and its events already packed if they are."""
        if not self._resolving or self._described is not None:
            return self.describe(**played), None
        if across_slice is not None:
            # A packing holds the event stream and not the table a pulse is
            # integrated over, so a profiled run walks the description instead
            # of rebinding onto a packing that has no table in it.
            return self.describe(**played), None
        if not isinstance(repetitions, int):
            # How many playings a settled run takes is decided against the
            # tissue it is given, so there is no one packing to hold fixed.
            return self.describe(**played), None
        where = target_device(tissue, device)
        settings = {
            "repetitions": repetitions,
            "record": record,
            "rf_raster_time_s": self.rf_raster_time_s,
        }
        key = run_key(played, device=where, **settings)
        if self._packing is not None and self._packing.matches(key):
            return self._packing.description, self._packing.pack(played)
        if any(key == refused for refused in self._refused):
            return self.describe(**played), None
        packing = bind(self, played, device=where, **settings)
        if packing is None:
            self._refused = (*self._refused, key)
            return self.describe(**played), None
        self._packing = packing
        return packing.description, packing.pack(played)

    def _split(self, values: Mapping[str, Any]) -> tuple[dict, dict]:
        """Tell the property arguments from the sequence ones.

        The protocol the constructor fixed joins the sequence arguments here,
        so a closed form reads everything it was written with from what it is
        handed, whichever side gave it. A call naming one again wins.
        """
        _no_renamed(values)
        held, sequence = super()._split(values)
        return held, {**self.protocol, **sequence}

    def _backend(self, values: Mapping[str, Any]) -> Any:
        """Return the caller's array library, from the call or the constructor.

        A call carrying arrays of its own decides, even when they are torch --
        the tissue is what the answer is about. Only a call with no arrays at
        all falls back to what the simulator was built from.
        """
        if any(is_array(value) for value in values.values()):
            return super()._backend(values)
        return self._brought

    # -- what a protocol says -----------------------------------------------

    def layout(self, **protocol: Any) -> Sequence[Operator | tuple[Any, Operator]]:
        """Return the operators of one repetition, in the order they play.

        A bare operator starts where the one before it ended; one given as
        ``(offset_s, operator)`` starts that far into the repetition instead,
        which is how a sequence that times itself from an echo says so.

        Raises
        ------
        NotImplementedError
            If the subclass implements neither this nor :meth:`describe`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} implements neither layout() nor describe()"
        )

    def repetition_s(self, played_s: Any, **protocol: Any) -> Any:
        """Return how long one repetition lasts, given what the layout played.

        The default is the span the layout covers. A sequence whose TR is
        longer than what it plays -- a refocused train waiting out its
        recovery -- says so here, and only a run of more than one repetition
        can tell the difference.
        """
        del protocol
        return played_s

    def played(self, **sequence: Any) -> dict[str, Any]:
        """Return the protocol as it will be laid out.

        The constructor's arguments, with anything given at the call
        overriding them, every array read as torch. Anything naming a run
        setting is left out: those describe the run, and a layout has no use
        for them.
        """
        given = {
            name: value for name, value in sequence.items() if name not in RUN_SETTINGS
        }
        return {**self.protocol, **read(given)}

    @property
    def rf_raster_time_s(self) -> float:
        """The dwell this simulator's RF shapes are sampled on."""
        if self._described is not None:
            return self._described.rf_raster_time_s
        return _RF_RASTER_TIME_S

    def describe(self, **protocol: Any) -> SequenceDescription:
        """Return the description this protocol plays."""
        if self._described is not None:
            return self._described
        events, played_s = compose(*self.layout(**protocol))
        return SequenceDescription(
            subsequence_index=0,
            tr_duration_us=1e6 * self.repetition_s(played_s, **protocol),
            events=events,
            rf_definitions=dict(self.model.definitions),
            shim_definitions=dict(self.shims),
            crusher_dephasing_rad=self.crusher_dephasing_rad,
            voxel_size_m=self.voxel_size_m,
        )

    @classmethod
    def from_pulseq(
        cls,
        path: Any,
        *,
        tr_index: int | None = None,
        **settings: Any,
    ) -> Simulator:
        """Return a simulator over one repetition of a Pulseq ``.seq`` file.

        The offline half of :meth:`from_description`: the same events, read
        from the file a scanner would be given rather than from the stream it
        sends back. The file states how many blocks a repetition holds, so
        nothing here searches for the period; naming the simulator is what says
        how those events are to be played.

        Parameters
        ----------
        path : str or Path
            The sequence file.
        tr_index : int, optional
            Which repetition to read, counted in whole repetitions. Defaults to
            the file's ``TRRef`` definition, and otherwise to the first
            repetition that acquires -- which is refused when the repetitions
            differ in their pulses.
        settings : Any
            Run settings and tissue, as the constructor takes them.

        Returns
        -------
            A simulator playing that repetition.
        """
        from ..sequence._pulseq import read_pulseq_description

        return cls.from_description(
            read_pulseq_description(path, tr_index=tr_index), **settings
        )

    @classmethod
    def from_description(
        cls,
        description: SequenceDescription,
        model: SpinPhysics | None = None,
        **settings: Any,
    ) -> Simulator:
        """Return a simulator over a stream someone else assembled.

        This is the path a description arriving from a scanner takes:
        ``FSESimulator.from_description(stream)`` says the events are to be
        read as a refocused train, and the only thing left to give is the
        tissue. Echo spacing, echo train length, flip angles and pulse shapes
        are in the stream and are not named again.

        Which simulator you call it on is the whole of what you choose, and it
        matters. A description says what was played -- an RF pulse, tagged with
        the use its designer gave it, and an ADC window -- and says nothing
        about the gradients between them, because the transport carries none.
        The dephasing lives in the handlers instead: a
        :func:`~torchsim.SSFPFidReadout` winds one order after every sample, a
        :func:`~torchsim.SSFPEchoReadout` winds it before, a
        :func:`~torchsim.SPGRReadout` spoils, and a refocusing pulse is
        crushed either side. So the events are re-emitted through this model's
        own operators rather than taken as they arrive.

        Parameters
        ----------
        description : SequenceDescription
            The stream, as the MRD client decodes it or a Pulseq design
            exports it.
        model : SpinPhysics, optional
            The physics to read it with. Defaults to this simulator's own,
            which is what naming a concrete one is for.
        settings : Any, optional
            Run settings and tissue, as the constructor takes them.

        Raises
        ------
        ValueError
            If called on :class:`Simulator` itself, which names no handlers and
            so says nothing about how the stream is to be read.
        """
        if cls is Simulator and model is None:
            raise ValueError(
                "from_description reads a stream through a simulator's own "
                "handlers, so call it on the simulator whose sequence it is -- "
                "FSESimulator.from_description(...) for a refocused train"
            )
        physics = model if model is not None else cls.model
        simulator = _Described(model=physics, **settings)
        simulator._described = realised(description, physics)
        return simulator

    # -- what a signal model owes -------------------------------------------

    def evaluate(self, properties: Mapping[str, Any], **sequence: Any) -> torch.Tensor:
        """Run one simulation of the described protocol.

        ``states``, ``repetitions``, ``record``, ``device`` and ``execution``
        describe the run and are taken here, each falling back to what the
        constructor was given; everything else overrides a protocol argument.
        """
        given = dict(sequence)
        states = given.pop("states", self.states)
        settings = {
            "repetitions": given.pop("repetitions", self.repetitions),
            "record": given.pop("record", self.record),
            "device": given.pop("device", None),
            "rf_raster_time_s": self.rf_raster_time_s,
        }
        target = given.pop("execution", self.execution)
        profile = across_the_slice(given.pop("across_slice", None)) or self.across_slice
        played = self.played(**given)
        tissue = self.model.tissue(properties)
        described, events = self._structure(
            played,
            tissue,
            repetitions=settings["repetitions"],
            record=settings["record"],
            device=settings["device"],
            across_slice=profile,
        )
        block = nullcontext() if target is None else execution(target)
        with block:
            return (
                EpgEngine()
                .simulate(
                    described,
                    tissue,
                    nstates=states,
                    events=events,
                    across_slice=profile,
                    **settings,
                )
                .signal
            )


class _Described(Simulator):
    """A simulator whose description was handed to it whole."""
