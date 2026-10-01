"""Reading the sequence description an MRD stream carries, as Pulserver's proxy sends it.

The stream is written here the way the proxy writes it -- the XML header, one
``TEXT`` message, then the first acquisition -- from a spoiled gradient echo
built with pypulseq, and the description read back is checked against the rows
it was written from and against the spoiled gradient-echo steady state, which
has never seen a stream.
"""

from __future__ import annotations

import base64
import io
import json
import math
import struct

import numpy as np
import pytest
import torch

from torchsim import Excitation, SPGRReadout
from torchsim.model import Simulator
from torchsim.sequence import EventType, RfUse, read_mrd_description
from torchsim.simulators import SPGRSimulator

pp = pytest.importorskip("pypulseq", reason="building the sequence needs pypulseq")

FLIP_DEG = 30.0
TR_S = 20e-3
SAMPLES, DWELL_S = 64, 10e-6
HEADER = "<ismrmrdHeader/>"
ACQUISITION = 1008


def _gradient_echo():
    system = pp.Opts(rf_dead_time=100e-6, rf_ringdown_time=20e-6, adc_dead_time=10e-6)
    sequence = pp.Sequence(system=system)
    pulse = pp.make_block_pulse(
        flip_angle=math.radians(FLIP_DEG),
        duration=200e-6,
        delay=system.rf_dead_time,
        system=system,
        use="excitation",
    )
    adc = pp.make_adc(
        num_samples=SAMPLES, dwell=DWELL_S, delay=system.adc_dead_time, system=system
    )
    sequence.add_block(pulse)
    sequence.add_block(adc)
    played = pp.calc_duration(pulse) + pp.calc_duration(adc)
    sequence.add_block(pp.make_delay(TR_S - played))
    return sequence


def _packed(values, dtype):
    return base64.b64encode(np.asarray(values, dtype=dtype).tobytes()).decode("ascii")


def _rows(sequence):
    """One row per block as the proxy lays it out, and the pulse it names.

    The envelope is normalised to a peak of one and played at the peak in Hz;
    its sample times are in raster steps.
    """
    raster_s = float(sequence.system.rf_raster_time)
    kinds, times_us, params = [], [], []
    start_s = 0.0
    definitions = []
    for index in range(1, len(sequence.block_durations) + 1):
        block = sequence.get_block(index)
        row = [0.0] * 7
        if block.rf is not None:
            signal = np.asarray(block.rf.signal)
            peak = float(np.max(np.abs(signal)))
            kinds.append(int(EventType.RF))
            times_us.append(1e6 * (start_s + block.rf.delay + block.rf.center))
            row[:3] = [len(definitions), int(RfUse.EXCITATION), peak]
            definitions.append(
                {
                    "id": len(definitions),
                    "bandwidth_hz": 0.0,
                    "num_bands": 1,
                    "band_frequency_offsets_hz": [0.0] * 8,
                    "band_bandwidth_hz": 0.0,
                    "magnitude": _shape(np.abs(signal) / peak),
                    "phase": _shape(np.angle(signal) / (2.0 * np.pi)),
                    "time": _shape(np.asarray(block.rf.t) / raster_s),
                }
            )
        elif block.adc is not None:
            kinds.append(int(EventType.ADC))
            times_us.append(
                1e6 * (start_s + block.adc.delay + (SAMPLES // 2) * block.adc.dwell)
            )
            row[:3] = [1, float(block.adc.phase_offset), 1]
        else:
            kinds.append(int(EventType.WAIT))
            times_us.append(1e6 * start_s)
        params.append(row)
        start_s += float(sequence.block_durations[index])
    return {
        "subsequence_index": 0,
        "tr_duration_us": 1e6 * start_s,
        "rf_raster_time_s": raster_s,
        "type": _packed(kinds, "<i4"),
        "timestamp_us": _packed(times_us, "<f8"),
        "params": _packed(params, "<f4"),
        "rf_definitions": definitions,
    }


def _shape(samples):
    return {"num_uncompressed": len(samples), "samples": _packed(samples, "<f4")}


def _message(identifier, text):
    body = (text + "\0").encode()
    return struct.pack("<HI", identifier, len(body)) + body


def _stream(subsequences):
    """The opening of a series as the proxy sends it."""
    text = json.dumps({"pulserver_sequence_description": subsequences})
    return io.BytesIO(
        _message(3, HEADER) + _message(5, text) + struct.pack("<H", ACQUISITION)
    )


@pytest.fixture
def written():
    return _rows(_gradient_echo())


def test_a_written_stream_reads_back_into_the_rows_it_was_built_from(written):
    stream = _stream([written])
    header, (described,) = read_mrd_description(stream)

    assert header == HEADER
    assert struct.unpack("<H", stream.read(2)) == (ACQUISITION,)
    kinds = np.frombuffer(base64.b64decode(written["type"]), "<i4")
    times_us = np.frombuffer(base64.b64decode(written["timestamp_us"]), "<f8")
    params = np.frombuffer(base64.b64decode(written["params"]), "<f4").reshape(-1, 7)
    assert [int(event.type) for event in described.events] == kinds.tolist()
    np.testing.assert_array_equal(
        [event.timestamp_us for event in described.events], times_us
    )
    for event, row in zip(described.events, params, strict=True):
        np.testing.assert_array_equal(
            np.asarray(event.params, dtype=np.float32), row[: len(event.params)]
        )
    assert described.tr_duration_us == written["tr_duration_us"]
    (pulse,) = written["rf_definitions"]
    for name in ("magnitude", "phase", "time"):
        np.testing.assert_array_equal(
            getattr(described.rf_definitions[0], name).samples,
            np.frombuffer(base64.b64decode(pulse[name]["samples"]), "<f4"),
        )


def test_every_file_of_a_chain_is_read_in_play_order(written):
    _, described = read_mrd_description(
        _stream([written, {**written, "subsequence_index": 1}])
    )
    assert [each.subsequence_index for each in described] == [0, 1]


def test_a_stream_whose_data_arrives_first_is_refused():
    stream = io.BytesIO(_message(3, HEADER) + struct.pack("<H", ACQUISITION))
    with pytest.raises(ValueError, match="before the sequence description"):
        read_mrd_description(stream)


def test_the_pulse_reads_back_at_the_angle_it_was_designed_for(written):
    """Peak-normalised and played at its peak, the envelope integrates to the flip."""
    _, (described,) = read_mrd_description(_stream([written]))
    excitation = described.events[0]
    flip_rad, _ = described.rf_definitions[0].flip_angle(
        excitation.rf_amplitude_hz, rf_raster_time_s=described.rf_raster_time_s
    )
    assert math.degrees(float(flip_rad)) == pytest.approx(FLIP_DEG, rel=1e-3)


def test_a_described_sequence_drives_the_spoiled_gradient_echo_steady_state(written):
    """The model is chosen here, by hand: the stream says what played, not what it means."""

    class Gre(Simulator):
        excitation = Excitation
        readout = SPGRReadout
        states = 1

    _, (described,) = read_mrd_description(_stream([written]))
    echo_time_ms = 1e-3 * (
        described.events[1].timestamp_us - described.events[0].timestamp_us
    )
    played = Gre.from_description(described, states=1, repetitions=400).simulate(
        T1=torch.tensor([830.0]), T2=torch.tensor([80.0]), M0=1.0
    )
    closed = SPGRSimulator(flip=FLIP_DEG, TR=TR_S * 1e3, TE=echo_time_ms).simulate(
        T1=torch.tensor([830.0]), T2star=torch.tensor([80.0]), M0=1.0
    )

    assert float(played.abs()) == pytest.approx(float(closed.abs()), rel=1e-3)
