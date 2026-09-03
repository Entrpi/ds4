#!/usr/bin/env python3
"""Rewrite a GGUF with one or more named tensors re-quantized, everything else
copied byte-for-byte (metadata, tensor order, every other tensor).

Built for the DSpark drafter Markov-table A/B: the ship drafter stores
markov_w2 as F16 and the on-device Markov refine reads all of it four times
per draft block, so a Q8_0 copy halves that traffic.  Isolating the change to
ONE tensor keeps the A/B honest (same experts, same heads, same metadata).

Usage:
  gguf_requant_tensor.py --in A.gguf --out B.gguf \
      --requant dspark.markov_w2.weight=q8_0 [--requant NAME=TYPE ...] [--verify]

--verify re-opens both files and proves: same metadata (minus alignment
bookkeeping), same tensor names/order/shapes, byte-identical data for every
tensor NOT listed in --requant, and the listed ones carry the requested type.
Exit 0 only when every check passes.  Needs gguf-py >= 0.10 (GGUFReader/Writer).
"""
import argparse, hashlib, sys
import numpy as np
import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType
from gguf import GGMLQuantizationType as QT
import gguf.quants as gq

QMAP = {"q8_0": QT.Q8_0, "f16": QT.F16, "f32": QT.F32, "bf16": QT.BF16}
SKIP_KEYS = {"general.architecture", "general.alignment"}  # writer owns these


def field_value(field):
    """Python value of a ReaderField (contents() on new gguf-py, manual fallback)."""
    if hasattr(field, "contents"):
        return field.contents()
    vt = field.types[0]
    if vt == GGUFValueType.ARRAY:
        sub = field.types[1]
        if sub == GGUFValueType.STRING:
            return [bytes(field.parts[i]).decode("utf-8") for i in field.data]
        return [field.parts[i][0].item() for i in field.data]
    if vt == GGUFValueType.STRING:
        return bytes(field.parts[field.data[0]]).decode("utf-8")
    return field.parts[field.data[0]][0].item()


def copy_metadata(reader, writer):
    arch = None
    copied = []
    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue
        if key == "general.architecture":
            arch = field_value(field)
            continue
        if key in SKIP_KEYS:
            continue
        vt = field.types[0]
        val = field_value(field)
        if vt == GGUFValueType.ARRAY:
            writer.add_key_value(key, val, GGUFValueType.ARRAY, sub_type=field.types[1])
        else:
            writer.add_key_value(key, val, vt)
        copied.append(key)
    return arch, copied


def tensor_sha(t):
    return hashlib.sha256(np.ascontiguousarray(t.data).tobytes()).hexdigest()


def rewrite(args):
    plan = {}
    for spec in args.requant:
        name, _, ty = spec.partition("=")
        if not name or ty.lower() not in QMAP:
            sys.exit(f"bad --requant '{spec}' (want NAME=TYPE, TYPE in {sorted(QMAP)})")
        plan[name] = QMAP[ty.lower()]

    r = GGUFReader(args.inp)
    names = [t.name for t in r.tensors]
    missing = [n for n in plan if n not in names]
    if missing:
        sys.exit(f"tensors not in {args.inp}: {missing}")

    arch = field_value(r.fields["general.architecture"])
    w = GGUFWriter(args.out, arch, use_temp_file=True)
    _, copied = copy_metadata(r, w)
    print(f"metadata: arch={arch} copied {len(copied)} keys", flush=True)

    for t in r.tensors:
        ttype = t.tensor_type
        if t.name in plan:
            target = plan[t.name]
            if ttype not in (QT.F16, QT.F32, QT.BF16):
                sys.exit(f"{t.name}: source type {ttype.name} is quantized; only F16/F32/BF16 sources are re-quantized")
            f32 = np.ascontiguousarray(t.data).astype(np.float32)
            if target == QT.F32:
                w.add_tensor(t.name, f32)
            elif target == QT.F16:
                w.add_tensor(t.name, f32.astype(np.float16))
            else:
                q = gq.quantize(f32, target)          # uint8 [..., bytes_per_row]
                w.add_tensor(t.name, q, raw_dtype=target)
            print(f"  ~ {t.name:34s} {ttype.name:7s} -> {target.name:7s} {tuple(f32.shape)}", flush=True)
        else:
            data = np.ascontiguousarray(t.data)
            if data.dtype == np.uint8 and ttype not in (QT.F16, QT.F32, QT.BF16):
                w.add_tensor(t.name, data, raw_dtype=ttype)   # byte-shaped, writer derives dims
            else:
                w.add_tensor(t.name, data, raw_dtype=ttype)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()
    print(f"wrote {args.out}", flush=True)
    return plan


def verify(args, plan):
    a = GGUFReader(args.inp)
    b = GGUFReader(args.out)
    ok = True
    ka = {k: field_value(f) for k, f in a.fields.items() if not k.startswith("GGUF.") and k not in SKIP_KEYS}
    kb = {k: field_value(f) for k, f in b.fields.items() if not k.startswith("GGUF.") and k not in SKIP_KEYS}
    if ka != kb:
        ok = False
        print(f"VERIFY metadata differs: only-in-src={set(ka)-set(kb)} only-in-out={set(kb)-set(ka)} "
              f"changed={[k for k in ka if k in kb and ka[k]!=kb[k]]}")
    else:
        print(f"VERIFY metadata identical ({len(ka)} keys)")
    if [t.name for t in a.tensors] != [t.name for t in b.tensors]:
        ok = False
        print("VERIFY tensor name/order differs")
    else:
        print(f"VERIFY tensor order identical ({len(a.tensors)} tensors)")
    tb = {t.name: t for t in b.tensors}
    same = changed = 0
    for ta in a.tensors:
        tt = tb.get(ta.name)
        if tt is None:
            ok = False; print(f"VERIFY missing in out: {ta.name}"); continue
        if list(ta.shape) != list(tt.shape):
            ok = False; print(f"VERIFY shape differs: {ta.name} {list(ta.shape)} vs {list(tt.shape)}")
        if ta.name in plan:
            if tt.tensor_type != plan[ta.name]:
                ok = False; print(f"VERIFY {ta.name}: type {tt.tensor_type.name}, wanted {plan[ta.name].name}")
            else:
                changed += 1
                print(f"VERIFY {ta.name}: {ta.tensor_type.name} -> {tt.tensor_type.name} n_bytes {ta.n_bytes} -> {tt.n_bytes}")
        else:
            if ta.tensor_type != tt.tensor_type or tensor_sha(ta) != tensor_sha(tt):
                ok = False; print(f"VERIFY data differs: {ta.name}")
            else:
                same += 1
    print(f"VERIFY {same} tensors byte-identical, {changed} re-quantized -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--requant", action="append", default=[], help="NAME=TYPE (q8_0|f16|f32|bf16)")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if not args.requant:
        sys.exit("nothing to do: pass at least one --requant NAME=TYPE")
    plan = rewrite(args)
    if args.verify and not verify(args, plan):
        sys.exit(1)


if __name__ == "__main__":
    main()
