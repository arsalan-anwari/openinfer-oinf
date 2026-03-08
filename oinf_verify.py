"""Verify and pretty-print .oinf files."""
from __future__ import annotations

import argparse
import math
import struct
from typing import Any, Dict, List, Tuple

import numpy as np

from oinf_common import OinfError, align_up, read_string
from oinf_format import format_scalar, format_values, histogram_string
from oinf_numeric import bf16_to_f32, f8_to_f32, f8_to_f32_scalar
from oinf_packed import unpack_signed_bits, unpack_t1_bits, unpack_unsigned_bits
from oinf_types import PACKED_BITS_PER, ValueType, VT_NAME, VT_SIZE, VT_TO_DTYPE, tensor_nbytes


def _tensor_stats(values: np.ndarray) -> Dict[str, Any]:
    """Compute summary stats for a tensor."""
    if values.size == 0:
        return {
            "numel": 0,
            "nbytes": int(values.nbytes),
            "min": math.nan,
            "max": math.nan,
            "mean": math.nan,
            "median": math.nan,
            "std": math.nan,
        }
    numeric = values.astype(np.float64) if values.dtype == np.bool_ else values.astype(np.float64, copy=False)
    stats = {
        "numel": int(values.size),
        "nbytes": int(values.nbytes),
        "min": float(np.min(numeric)),
        "max": float(np.max(numeric)),
        "mean": float(np.mean(numeric)),
        "median": float(np.median(numeric)),
        "std": float(np.std(numeric)),
    }
    return stats


def _parse_metadata_value(blob: bytes, entry: Dict[str, Any]) -> Any:
    """Parse a metadata entry payload into a Python value."""
    vtype = entry["value_type"]
    offset = entry["value_offset"]
    nbytes = entry["value_nbytes"]
    if offset + nbytes > len(blob):
        raise OinfError("Metadata value exceeds file bounds")
    payload = blob[offset : offset + nbytes]
    if vtype == ValueType.STRING:
        text, next_offset = read_string(payload, 0)
        if next_offset != nbytes:
            raise OinfError("STRING payload size mismatch")
        return text
    if vtype == ValueType.BOOL:
        if nbytes != 1:
            raise OinfError("BOOL payload size mismatch")
        return payload[0] != 0
    if vtype == ValueType.BF16:
        if nbytes != 2:
            raise OinfError("BF16 payload size mismatch")
        bits = np.frombuffer(payload, dtype=np.dtype(np.uint16).newbyteorder("<"))
        return bf16_to_f32(bits)[0].item()
    if vtype == ValueType.F8:
        if nbytes != 1:
            raise OinfError("F8 payload size mismatch")
        return f8_to_f32_scalar(payload[0])
    if vtype in PACKED_BITS_PER:
        bits_per = PACKED_BITS_PER[vtype]
        if nbytes != 1:
            raise OinfError("Packed int payload size mismatch")
        if vtype in (ValueType.U4, ValueType.U2, ValueType.U1):
            return int(unpack_unsigned_bits(payload, bits_per, 1)[0])
        if vtype == ValueType.T1:
            return int(unpack_t1_bits(payload, 1)[0])
        if vtype == ValueType.T2:
            return int(unpack_signed_bits(payload, bits_per, 1)[0])
        return int(unpack_signed_bits(payload, bits_per, 1)[0])
    if vtype in VT_SIZE:
        expected = VT_SIZE[vtype]
        if nbytes != expected:
            raise OinfError("Metadata scalar size mismatch")
        dtype = np.dtype(VT_TO_DTYPE[vtype]).newbyteorder("<")
        return np.frombuffer(payload, dtype=dtype)[0].item()
    if vtype == ValueType.BITSET:
        if nbytes < 8:
            raise OinfError("BITSET payload too small")
        bit_count, byte_count = struct.unpack_from("<II", payload, 0)
        expected_bytes = (bit_count + 7) // 8
        if byte_count != expected_bytes:
            raise OinfError("BITSET byte_count mismatch")
        total = 8 + byte_count
        total = align_up(total)
        if nbytes != total:
            raise OinfError("BITSET payload size mismatch")
        data = payload[8 : 8 + byte_count]
        return {"bit_count": bit_count, "bytes": data}
    if vtype == ValueType.NDARRAY:
        if nbytes < 8:
            raise OinfError("NDARRAY payload too small")
        element_type, ndim = struct.unpack_from("<II", payload, 0)
        if element_type not in VT_TO_DTYPE:
            raise OinfError("NDARRAY element type invalid")
        dims_offset = 8
        dims_size = 8 * ndim
        if dims_offset + dims_size > nbytes:
            raise OinfError("NDARRAY dims exceed payload")
        if ndim:
            dims = struct.unpack_from("<" + "Q" * ndim, payload, dims_offset)
        else:
            dims = ()
        data_offset = dims_offset + dims_size
        numel = int(np.prod(dims)) if dims else 1
        data_nbytes = tensor_nbytes(element_type, numel)
        total = align_up(data_offset + data_nbytes)
        if nbytes != total:
            raise OinfError("NDARRAY payload size mismatch")
        raw = payload[data_offset : data_offset + data_nbytes]
        if element_type == ValueType.BOOL:
            arr = np.frombuffer(raw, dtype=np.uint8).astype(np.bool_)
        elif element_type == ValueType.BF16:
            arr = bf16_to_f32(np.frombuffer(raw, dtype=np.dtype(np.uint16).newbyteorder("<")))
        elif element_type == ValueType.F8:
            arr = f8_to_f32(np.frombuffer(raw, dtype=np.uint8))
        elif element_type in PACKED_BITS_PER:
            bits_per = PACKED_BITS_PER[element_type]
            if element_type in (ValueType.U4, ValueType.U2, ValueType.U1):
                arr = unpack_unsigned_bits(raw, bits_per, numel)
            elif element_type == ValueType.T1:
                arr = unpack_t1_bits(raw, numel)
            else:
                arr = unpack_signed_bits(raw, bits_per, numel)
        else:
            arr = np.frombuffer(raw, dtype=np.dtype(VT_TO_DTYPE[element_type]).newbyteorder("<"))
        return {"dtype": element_type, "dims": dims, "array": arr.reshape(dims)}
    raise OinfError(f"Unsupported metadata value type {vtype}")


def _parse_quant_payload(blob: bytes, tensor: Dict[str, Any], offset_data: int, file_size: int) -> Dict[str, Any]:
    """Parse and validate tensor quant payload."""
    q_offset = tensor["quant_offset"]
    q_nbytes = tensor["quant_nbytes"]
    if q_offset % 8 != 0:
        raise OinfError("Tensor quant offset not aligned")
    if q_offset < offset_data:
        raise OinfError("Tensor quant offset precedes data section")
    if q_offset + q_nbytes > file_size:
        raise OinfError("Tensor quant payload exceeds file size")
    if q_nbytes < 48:
        raise OinfError("Tensor quant payload too small")
    payload = blob[q_offset : q_offset + q_nbytes]

    scheme, scale_mode, zp_mode, reserved, scale_axis, scale_count, zp_axis, zp_count = struct.unpack_from(
        "<IIIIQQQQ", payload, 0
    )
    if reserved != 0:
        raise OinfError("Tensor quant reserved must be 0")
    if scheme not in (1, 2):
        raise OinfError("Tensor quant scheme invalid")
    if scale_mode not in (1, 2):
        raise OinfError("Tensor quant scale mode invalid")
    if zp_mode not in (0, 1, 2):
        raise OinfError("Tensor quant zero_point mode invalid")

    scale_bytes = scale_count * 4
    zp_bytes = zp_count * 4
    expected = align_up(48 + scale_bytes + zp_bytes)
    if q_nbytes != expected:
        raise OinfError("Tensor quant payload size mismatch")
    scale_start = 48
    scale_end = scale_start + scale_bytes
    zp_end = scale_end + zp_bytes
    scale_values = np.frombuffer(payload[scale_start:scale_end], dtype=np.dtype(np.float32).newbyteorder("<"))
    zp_values = np.frombuffer(payload[scale_end:zp_end], dtype=np.dtype(np.int32).newbyteorder("<"))

    dims = tensor["dims"]
    ndim = tensor["ndim"]
    if scale_mode == 1:
        if scale_axis != 0 or scale_count != 1:
            raise OinfError("Per-tensor scale must use axis=0 and count=1")
    else:
        if scale_axis >= ndim:
            raise OinfError("Per-channel scale axis out of range")
        if scale_count != dims[scale_axis]:
            raise OinfError("Per-channel scale count mismatch")

    if zp_mode == 0:
        if zp_count != 0:
            raise OinfError("No-zero-point mode requires count=0")
    elif zp_mode == 1:
        if scale_mode != 1:
            raise OinfError("Per-tensor zero-point requires per-tensor scale")
        if zp_axis != 0 or zp_count != 1:
            raise OinfError("Per-tensor zero-point must use axis=0 and count=1")
    else:
        if scale_mode != 2:
            raise OinfError("Per-channel zero-point requires per-channel scale")
        if zp_axis != scale_axis:
            raise OinfError("Per-channel zero-point axis must match scale axis")
        if zp_axis >= ndim:
            raise OinfError("Per-channel zero-point axis out of range")
        if zp_count != dims[zp_axis]:
            raise OinfError("Per-channel zero-point count mismatch")

    if scheme == 1 and zp_mode != 0:
        raise OinfError("Symmetric quantization cannot include zero_point")

    return {
        "scheme": "symmetric" if scheme == 1 else "asymmetric",
        "scale_mode": "per_tensor" if scale_mode == 1 else "per_channel",
        "scale_axis": int(scale_axis),
        "scale_values": scale_values,
        "zero_point_mode": {0: "none", 1: "per_tensor", 2: "per_channel"}[zp_mode],
        "zero_point_axis": int(zp_axis),
        "zero_point_values": zp_values,
    }


def parse_file(path: str) -> None:
    """Parse and print validation output for a .oinf file."""
    with open(path, "rb") as handle:
        blob = handle.read()

    if len(blob) < 69:
        raise OinfError("File too small for header")
    header = struct.unpack_from("<5sIIIIIIQQQQQ", blob, 0)
    magic = header[0]
    if magic != b"OINF\x00":
        raise OinfError("Bad magic")
    version = header[1]
    if version != 2:
        raise OinfError(f"Unsupported version {version}")
    n_sizevars, n_metadata, n_tensors = header[3], header[4], header[5]
    offset_sizevars, offset_metadata, offset_tensors, offset_data, file_size = header[7:]
    if file_size != len(blob):
        raise OinfError("File size mismatch")
    offsets = [offset_sizevars, offset_metadata, offset_tensors, offset_data, file_size]
    if offsets != sorted(offsets):
        raise OinfError("Offsets are not ascending")
    for off in offsets[:-1]:
        if off % 8 != 0:
            raise OinfError("Section offset not 8-byte aligned")
        if off > file_size:
            raise OinfError("Section offset exceeds file size")

    cursor = offset_sizevars
    sizevars = []
    names = set()
    for _ in range(n_sizevars):
        name, cursor = read_string(blob, cursor)
        if name in names:
            raise OinfError(f"Duplicate sizevar '{name}'")
        names.add(name)
        if cursor + 8 > offset_metadata:
            raise OinfError("Sizevars table exceeds metadata offset")
        value = struct.unpack_from("<Q", blob, cursor)[0]
        cursor += 8
        sizevars.append((name, value))

    cursor = offset_metadata
    metadata_entries = []
    names = set()
    for _ in range(n_metadata):
        key, cursor = read_string(blob, cursor)
        if key in names:
            raise OinfError(f"Duplicate metadata key '{key}'")
        names.add(key)
        if cursor + 24 > offset_tensors:
            raise OinfError("Metadata table exceeds tensor offset")
        value_type, flags, value_nbytes, value_offset = struct.unpack_from("<IIQQ", blob, cursor)
        cursor += 24
        if flags != 0:
            raise OinfError("Metadata flags must be 0")
        if value_offset % 8 != 0:
            raise OinfError("Metadata value offset not aligned")
        if value_offset < offset_data:
            raise OinfError("Metadata value offset precedes data section")
        if value_offset + value_nbytes > file_size:
            raise OinfError("Metadata value exceeds file size")
        if value_type not in VT_NAME:
            raise OinfError(f"Metadata value type {value_type} invalid")
        metadata_entries.append(
            {
                "key": key,
                "value_type": value_type,
                "value_nbytes": value_nbytes,
                "value_offset": value_offset,
            }
        )

    cursor = offset_tensors
    tensors = []
    names = set()
    for _ in range(n_tensors):
        name, cursor = read_string(blob, cursor)
        if name in names:
            raise OinfError(f"Duplicate tensor name '{name}'")
        names.add(name)
        if cursor + 12 > offset_data:
            raise OinfError("Tensor table exceeds data offset")
        dtype, ndim, flags = struct.unpack_from("<III", blob, cursor)
        cursor += 12
        if dtype not in VT_SIZE or dtype in (ValueType.BITSET, ValueType.STRING, ValueType.NDARRAY):
            raise OinfError("Invalid tensor dtype")
        dims = ()
        if ndim > 0:
            dims = struct.unpack_from("<" + "Q" * ndim, blob, cursor)
            cursor += 8 * ndim
        data_nbytes, data_offset, quant_nbytes, quant_offset = struct.unpack_from("<QQQQ", blob, cursor)
        cursor += 32
        has_data = flags & 1
        has_quant = (flags & 2) != 0
        if flags & ~0x3:
            raise OinfError("Tensor flags contain unsupported bits")
        if not has_data:
            if data_nbytes != 0 or data_offset != 0:
                raise OinfError("Tensor without data must have zero offset/size")
        else:
            if data_offset % 8 != 0:
                raise OinfError("Tensor data offset not aligned")
            if data_offset < offset_data:
                raise OinfError("Tensor data offset precedes data section")
            numel = int(np.prod(dims)) if dims else 1
            expected = tensor_nbytes(dtype, numel)
            if data_nbytes != expected:
                raise OinfError("Tensor data_nbytes mismatch")
            if data_offset + data_nbytes > file_size:
                raise OinfError("Tensor data exceeds file size")
        if not has_quant:
            if quant_nbytes != 0 or quant_offset != 0:
                raise OinfError("Tensor without quant must have zero quant offset/size")
        else:
            if quant_offset % 8 != 0:
                raise OinfError("Tensor quant offset not aligned")
            if quant_offset < offset_data:
                raise OinfError("Tensor quant offset precedes data section")
            if quant_offset + quant_nbytes > file_size:
                raise OinfError("Tensor quant payload exceeds file size")
        tensors.append(
            {
                "name": name,
                "dtype": dtype,
                "ndim": ndim,
                "dims": dims,
                "has_data": bool(has_data),
                "data_nbytes": data_nbytes,
                "data_offset": data_offset,
                "has_quant": bool(has_quant),
                "quant_nbytes": quant_nbytes,
                "quant_offset": quant_offset,
            }
        )

    for name, value in sizevars:
        print(f"{name} := {value}")
    if sizevars:
        print()

    for entry in metadata_entries:
        value = _parse_metadata_value(blob, entry)
        key = entry["key"]
        vtype = entry["value_type"]
        if vtype == ValueType.NDARRAY:
            dtype = VT_NAME[value["dtype"]]
            dims = ", ".join(str(d) for d in value["dims"])
            arr = value["array"]
            print(f"{key}: {dtype}[{dims}] = {format_values(arr)}")
        elif vtype == ValueType.BITSET:
            bit_count = value["bit_count"]
            data = value["bytes"]
            preview = " ".join(f"{b:02x}" for b in data[:4])
            print(f"{key}: bitset[{bit_count}] = {preview}")
        else:
            print(f"{key}: {VT_NAME[vtype]} = {format_scalar(value)}")

    if metadata_entries:
        print()

    for tensor in tensors:
        name = tensor["name"]
        dtype = VT_NAME[tensor["dtype"]]
        dims = ", ".join(str(d) for d in tensor["dims"])
        if tensor["has_data"]:
            data_offset = tensor["data_offset"]
            data_nbytes = tensor["data_nbytes"]
            raw = blob[data_offset : data_offset + data_nbytes]
            if tensor["dtype"] == ValueType.BOOL:
                arr = np.frombuffer(raw, dtype=np.uint8).astype(np.bool_)
            elif tensor["dtype"] == ValueType.BF16:
                arr = bf16_to_f32(np.frombuffer(raw, dtype=np.dtype(np.uint16).newbyteorder("<")))
            elif tensor["dtype"] == ValueType.F8:
                arr = f8_to_f32(np.frombuffer(raw, dtype=np.uint8))
            elif tensor["dtype"] in (ValueType.I4, ValueType.I2, ValueType.I1):
                bits_per = PACKED_BITS_PER[tensor["dtype"]]
                arr = unpack_signed_bits(
                    raw,
                    bits_per,
                    int(np.prod(tensor["dims"])) if tensor["dims"] else 1,
                )
            elif tensor["dtype"] in (ValueType.U4, ValueType.U2, ValueType.U1):
                bits_per = PACKED_BITS_PER[tensor["dtype"]]
                arr = unpack_unsigned_bits(
                    raw,
                    bits_per,
                    int(np.prod(tensor["dims"])) if tensor["dims"] else 1,
                )
            elif tensor["dtype"] == ValueType.T1:
                arr = unpack_t1_bits(
                    raw, int(np.prod(tensor["dims"])) if tensor["dims"] else 1
                )
            elif tensor["dtype"] == ValueType.T2:
                bits_per = PACKED_BITS_PER[tensor["dtype"]]
                arr = unpack_signed_bits(
                    raw,
                    bits_per,
                    int(np.prod(tensor["dims"])) if tensor["dims"] else 1,
                )
            else:
                arr = np.frombuffer(raw, dtype=np.dtype(VT_TO_DTYPE[tensor["dtype"]]).newbyteorder("<"))
            arr = arr.reshape(tensor["dims"]) if tensor["dims"] else arr.reshape(())
            if arr.size == 0:
                print(f"{name}: {dtype}[{dims}] -- uninitialized")
                print()
                continue
            if arr.size == 1:
                print(f"{name}: {dtype} = {format_scalar(arr.item())}")
                print()
                continue
            print(f"{name}: {dtype}[{dims}] = {format_values(arr)}")
            stats = _tensor_stats(arr)
            print(
                f"- [nbytes: {stats['nbytes']}, min: {stats['min']:.6g}, max: {stats['max']:.6g}, "
                f"mean: {stats['mean']:.6g}, median: {stats['median']:.6g}, std: {stats['std']:.6g}]"
            )
            print(f"- hist: {histogram_string(arr)}")
            if tensor["has_quant"]:
                quant = _parse_quant_payload(blob, tensor, offset_data, file_size)
                scale_vals = quant["scale_values"]
                if quant["scale_mode"] == "per_tensor":
                    scale_desc = f"per_tensor({float(scale_vals[0]):.6g})"
                else:
                    scale_desc = f"per_channel(axis={quant['scale_axis']}, count={len(scale_vals)})"
                zp_vals = quant["zero_point_values"]
                if quant["zero_point_mode"] == "none":
                    zp_desc = "none"
                elif quant["zero_point_mode"] == "per_tensor":
                    zp_desc = f"per_tensor({int(zp_vals[0])})"
                else:
                    zp_desc = f"per_channel(axis={quant['zero_point_axis']}, count={len(zp_vals)})"
                print(f"- quant: scheme={quant['scheme']}, scale={scale_desc}, zero_point={zp_desc}")
            print()
        else:
            print(f"{name}: {dtype}[{dims}] -- uninitialized")
            if tensor["has_quant"]:
                quant = _parse_quant_payload(blob, tensor, offset_data, file_size)
                scale_vals = quant["scale_values"]
                if quant["scale_mode"] == "per_tensor":
                    scale_desc = f"per_tensor({float(scale_vals[0]):.6g})"
                else:
                    scale_desc = f"per_channel(axis={quant['scale_axis']}, count={len(scale_vals)})"
                zp_vals = quant["zero_point_values"]
                if quant["zero_point_mode"] == "none":
                    zp_desc = "none"
                elif quant["zero_point_mode"] == "per_tensor":
                    zp_desc = f"per_tensor({int(zp_vals[0])})"
                else:
                    zp_desc = f"per_channel(axis={quant['zero_point_axis']}, count={len(zp_vals)})"
                print(f"- quant: scheme={quant['scheme']}, scale={scale_desc}, zero_point={zp_desc}")
            print()


def main() -> None:
    """CLI entry point for .oinf verification."""
    parser = argparse.ArgumentParser(description="Verify and pretty-print .oinf")
    parser.add_argument("path", help="Path to .oinf file")
    args = parser.parse_args()
    parse_file(args.path)


if __name__ == "__main__":
    main()
