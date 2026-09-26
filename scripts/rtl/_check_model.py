"""Scratch check: does model.forward(requant="float") reproduce the golden exactly?"""
import importlib.util
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from drivesentinel import config as C
from drivesentinel.rtl import model as M

spec = importlib.util.spec_from_file_location(
    "golden_reference", os.path.join(C.EXPORT_DIR, "golden_reference.py"))
golden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden)

exp = M.Export()
print("pool_index", exp.pool_index, "pool_length", exp.pool_length())
for i, m in enumerate(exp.multipliers()):
    print(f"  layer {i} M in [{m.min():.4e}, {m.max():.4e}]")

rng = np.random.default_rng(0)
x = rng.normal(0.75, 0.4, size=(64, 5, 512))

t = time.time()
g_log = golden.run_int8(x)
print("golden", time.time() - t, "s")

t = time.time()
r = M.forward(M.quantise_input(x, exp), exp, requant="float", trace=True)
print("model ", time.time() - t, "s")

print("max abs logit diff:", np.abs(g_log - r["logits"]).max())
print("argmax equal      :", bool((g_log.argmax(1) == r["pred"]).all()))
for i, a in enumerate(r["acc"]):
    print(f"  acc layer {i}: shape {a.shape} peak {int(np.abs(a).max()):,}")
