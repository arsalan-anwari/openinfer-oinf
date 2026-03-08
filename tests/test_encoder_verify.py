from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
import tempfile
import unittest
import struct

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oinf_encoder import (
    Bitset,
    QuantParams,
    QuantScale,
    QuantZeroPoint,
    ScalarValue,
    TensorSpec,
    UninitializedTensor,
    dataclass_to_oinf,
    write_oinf,
)
from oinf_common import OinfError, read_string
from oinf_verify import parse_file


@dataclasses.dataclass
class Sample:
    a: np.ndarray
    b: TensorSpec
    c: TensorSpec
    d: TensorSpec
    e: TensorSpec
    f: UninitializedTensor


@dataclasses.dataclass
class QuantModel:
    q: TensorSpec


class TestEncoderVerify(unittest.TestCase):
    def test_dataclass_roundtrip_with_verify(self) -> None:
        sample = Sample(
            a=np.array([[1, 2], [3, 4]], dtype=np.int16),
            b=TensorSpec(np.array([1.25, -2.0], dtype=np.float32), dtype="bf16"),
            c=TensorSpec(np.array([0.0, 2.0], dtype=np.float32), dtype="f8"),
            d=TensorSpec(np.array([-2, -1, 0, 1], dtype=np.int8), dtype="i4"),
            e=TensorSpec(np.array([0, 1, 1, 0], dtype=np.int8), dtype="u1"),
            f=UninitializedTensor("f32", (2, 2)),
        )
        sample.sizevars = {"B": 2}
        sample.metadata = {
            "title": "sample",
            "flag": True,
            "count": ScalarValue(3, dtype="u8"),
            "bitset": Bitset([True, False, True, True]),
            "meta_arr": [1, 2, 3],
        }

        payload = dataclass_to_oinf(sample)
        self.assertGreater(len(payload), 64)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.oinf"
            write_oinf(sample, str(path))
            parse_file(str(path))

    def _write_temp_payload(self, payload: bytes) -> Path:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "model.oinf"
        path.write_bytes(payload)
        return path

    def _tensor_quant_offset(self, payload: bytes) -> tuple[int, int]:
        header = struct.unpack_from("<5sIIIIIIQQQQQ", payload, 0)
        offset_tensors = int(header[9])
        cursor = offset_tensors
        _, cursor = read_string(payload, cursor)
        _, ndim, _ = struct.unpack_from("<III", payload, cursor)
        cursor += 12 + 8 * ndim
        _, _, quant_nbytes, quant_offset = struct.unpack_from("<QQQQ", payload, cursor)
        return int(quant_offset), int(quant_nbytes)

    def test_quant_per_tensor_fixture(self) -> None:
        model = QuantModel(
            q=TensorSpec(
                np.array([1, 2, 3, 4], dtype=np.uint8),
                dtype="u8",
                quant=QuantParams(
                    scheme="asymmetric",
                    scale=QuantScale.per_tensor(0.125),
                    zero_point=QuantZeroPoint.per_tensor(128),
                ),
            )
        )
        payload = dataclass_to_oinf(model)
        path = self._write_temp_payload(payload)
        parse_file(str(path))

    def test_quant_per_channel_fixture(self) -> None:
        model = QuantModel(
            q=TensorSpec(
                np.array([[1, 2], [3, 4]], dtype=np.int8),
                dtype="i8",
                quant=QuantParams(
                    scheme="asymmetric",
                    scale=QuantScale.per_channel(axis=0, values=[0.1, 0.2]),
                    zero_point=QuantZeroPoint.per_channel(axis=0, values=[0, 0]),
                ),
            )
        )
        payload = dataclass_to_oinf(model)
        path = self._write_temp_payload(payload)
        parse_file(str(path))

    def test_packed_non_quantized_fixture(self) -> None:
        model = QuantModel(
            q=TensorSpec(np.array([-2, -1, 0, 1], dtype=np.int8), dtype="i4")
        )
        payload = dataclass_to_oinf(model)
        path = self._write_temp_payload(payload)
        parse_file(str(path))

    def test_symmetric_with_zero_point_rejected(self) -> None:
        model = QuantModel(
            q=TensorSpec(
                np.array([1, 2, 3], dtype=np.int8),
                dtype="i8",
                quant=QuantParams(
                    scheme="symmetric",
                    scale=QuantScale.per_tensor(0.25),
                    zero_point=QuantZeroPoint.per_tensor(0),
                ),
            )
        )
        with self.assertRaises(OinfError):
            dataclass_to_oinf(model)

    def test_malformed_quant_axis_fails_verifier(self) -> None:
        model = QuantModel(
            q=TensorSpec(
                np.array([[1, 2], [3, 4]], dtype=np.int8),
                dtype="i8",
                quant=QuantParams(
                    scheme="asymmetric",
                    scale=QuantScale.per_channel(axis=0, values=[0.25, 0.5]),
                    zero_point=QuantZeroPoint.per_channel(axis=0, values=[0, 0]),
                ),
            )
        )
        payload = bytearray(dataclass_to_oinf(model))
        quant_offset, quant_nbytes = self._tensor_quant_offset(payload)
        self.assertGreaterEqual(quant_nbytes, 48)
        struct.pack_into("<Q", payload, quant_offset + 16, 5)
        path = self._write_temp_payload(bytes(payload))
        with self.assertRaises(OinfError):
            parse_file(str(path))

    def test_unsupported_legacy_version_rejected(self) -> None:
        model = QuantModel(q=TensorSpec(np.array([1, 2, 3], dtype=np.int8), dtype="i8"))
        payload = bytearray(dataclass_to_oinf(model))
        struct.pack_into("<I", payload, 5, 1)
        path = self._write_temp_payload(bytes(payload))
        with self.assertRaises(OinfError):
            parse_file(str(path))
