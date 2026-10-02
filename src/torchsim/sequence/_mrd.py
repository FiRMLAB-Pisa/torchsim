"""Reading the sequence description an MRD stream carries ahead of its data.

Pulserver's reconstruction proxy infers the description from the sequence file
a series was played from and sends it after the MRD XML header and before the
first acquisition, as one ``TEXT`` message: a JSON object under the key
``pulserver_sequence_description``, holding one entry per file of the sequence
chain. What is read here is mechanical; which signal model the rows drive is
chosen by whoever reads them.
"""

from __future__ import annotations

__all__ = ["read_mrd_description"]

import base64
import json
import struct
from typing import Any, BinaryIO

import numpy as np

from ._description import (
    AdcRole,
    EventType,
    RfDefinition,
    RfShape,
    RfUse,
    SequenceDescription,
    SequenceEvent,
    _trapezoid,
)

#: The key the description's JSON object opens with.
MESSAGE_KEY = "pulserver_sequence_description"

# MRD streaming message identifiers, and the fixed size of a FILENAME message.
_FILENAME, _CONFIG, _HEADER, _CLOSE, _TEXT = 1, 2, 3, 4, 5
_FILENAME_BYTES = 1024


def read_mrd_description(
    stream: BinaryIO,
) -> tuple[str | None, list[SequenceDescription]]:
    """Read an MRD stream up to its sequence description.

    The messages ahead of the description -- a config, a config file name, the
    XML header, other texts -- are read and the stream is left at the message
    after the description, which is where its first acquisition is.

    Parameters
    ----------
    stream : binary file-like
        The stream, read with ``read(n)``: an open file, or a socket's
        ``makefile("rb")``.

    Returns
    -------
    header : str or None
        The MRD XML header, ``None`` if the stream carried none before the
        description.
    descriptions : list of SequenceDescription
        One per file of the sequence chain, in play order. RF definitions carry
        the envelope normalised to a peak of one, played at the event's
        ``amplitude_hz``, and their sample times in units of the RF raster.

    Raises
    ------
    ValueError
        If any other message, or the end of the stream, comes before the
        description.
    """
    header = None
    while True:
        (identifier,) = struct.unpack("<H", _exactly(stream, 2))
        if identifier == _FILENAME:
            _exactly(stream, _FILENAME_BYTES)
        elif identifier in (_CONFIG, _HEADER, _TEXT):
            (length,) = struct.unpack("<I", _exactly(stream, 4))
            text = _exactly(stream, length).split(b"\0", 1)[0].decode("utf-8")
            if identifier == _HEADER:
                header = text
            elif identifier == _TEXT and text.startswith('{"' + MESSAGE_KEY + '"'):
                return header, [
                    _description(each) for each in json.loads(text)[MESSAGE_KEY]
                ]
        else:
            raise ValueError(
                f"MRD message {identifier} arrived before the sequence description"
            )


def _exactly(stream: BinaryIO, count: int) -> bytes:
    read = stream.read(count)
    if len(read) != count:
        raise ValueError("the MRD stream ended before the sequence description")
    return read


def _array(packed: str, dtype: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(packed), dtype=dtype)


def _description(fields: dict[str, Any]) -> SequenceDescription:
    kinds = _array(fields["type"], "<i4")
    times_us = _array(fields["timestamp_us"], "<f8")
    params = _array(fields["params"], "<f4").reshape(kinds.size, 7)
    raster_s = float(fields["rf_raster_time_s"])
    return SequenceDescription(
        subsequence_index=int(fields["subsequence_index"]),
        tr_duration_us=float(fields["tr_duration_us"]),
        events=tuple(
            _event(EventType(int(kind)), float(time_us), row)
            for kind, time_us, row in zip(kinds, times_us, params, strict=True)
        ),
        rf_definitions={
            int(each["id"]): _definition(each, raster_s)
            for each in fields["rf_definitions"]
        },
        rf_raster_time_s=raster_s,
    )


def _event(kind: EventType, time_us: float, row: np.ndarray) -> SequenceEvent:
    if kind is EventType.RF:
        return SequenceEvent.rf(
            time_us,
            definition_id=int(row[0]),
            use=RfUse(int(row[1])),
            amplitude_hz=float(row[2]),
            phase_rad=float(row[3]),
            frequency_hz=float(row[4]),
            shim_id=int(row[5]),
            slice_select_gradient_hz_per_m=float(row[6]),
        )
    if kind is EventType.ADC:
        return SequenceEvent.adc(
            time_us,
            role=AdcRole(int(row[0])),
            phase_rad=float(row[1]),
            is_echo=bool(row[2]),
        )
    return SequenceEvent.wait(time_us)


def _shape(fields: dict[str, Any] | None) -> RfShape | None:
    if fields is None:
        return None
    return RfShape(int(fields["num_uncompressed"]), _array(fields["samples"], "<f4"))


def _definition(fields: dict[str, Any], raster_s: float) -> RfDefinition:
    magnitude = _shape(fields["magnitude"])
    time = _shape(fields["time"])
    envelope = magnitude.decompress().astype(np.float64)
    time_s = (
        (np.arange(envelope.size) + 0.5) * raster_s
        if time is None
        else time.decompress(scale=raster_s).astype(np.float64)
    )
    return RfDefinition(
        id=int(fields["id"]),
        bandwidth_hz=float(fields["bandwidth_hz"]),
        num_bands=int(fields["num_bands"]),
        band_frequency_offsets_hz=tuple(
            float(value) for value in fields["band_frequency_offsets_hz"]
        ),
        band_bandwidth_hz=float(fields["band_bandwidth_hz"]),
        # Not carried by the stream: the stored envelope's squared integral.
        total_b1sq_power=float(_trapezoid(envelope**2, time_s)),
        magnitude=magnitude,
        phase=_shape(fields["phase"]),
        time=time,
    )
