#!/usr/bin/env python3
"""
Create a .oinf file for the accumulation demo example.

This model demonstrates user-defined accumulation types (acc attribute) for
matmul and sum_axis. Run this script first, then:

  cargo run --example accumulation_demo

The example uses the graph!{} DSL with acc=[f32, f32] to control accumulation
dtypes for matmul and sum_axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataclass_to_oinf import SizeVar, TensorSpec, write_oinf  # noqa: E402


@dataclass
class AccumulationDemoModel:
    B: SizeVar
    D: SizeVar
    H: SizeVar
    x: TensorSpec
    w: TensorSpec
    matmul_out: TensorSpec
    sum_axis_out: TensorSpec


def build_model() -> AccumulationDemoModel:
    rng = np.random.default_rng(42)
    B, D, H = 4, 8, 6
    x = rng.normal(scale=0.2, size=(B, D)).astype(np.float32)
    w = rng.normal(scale=0.2, size=(D, H)).astype(np.float32)
    matmul_out = x @ w
    sum_axis_out = matmul_out.sum(axis=1, keepdims=True).astype(np.float32)
    return AccumulationDemoModel(
        B=SizeVar(B),
        D=SizeVar(D),
        H=SizeVar(H),
        x=TensorSpec(x),
        w=TensorSpec(w),
        matmul_out=TensorSpec(matmul_out),
        sum_axis_out=TensorSpec(sum_axis_out),
    )


def main() -> None:
    model = build_model()
    # Write to openinfer-simulator/res/models so the simulator example can load it
    output_dir = ROOT / "res" / "models"
    output = output_dir / "accumulation_demo.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
