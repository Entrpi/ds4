#!/usr/bin/env python3
"""Extract the DeepSeek-V4-Flash-DSpark 3-stage MTP block-drafter from the HF
checkpoint (last 3 safetensors shards) into a ds4-format GGUF.

DSpark drafter = 3 full DeepSeek-V4 transformer layers (attn + 256-expert MoE,
dense/compress-ratio-0 SWA) whose KV is materialized from the *target's* fused
hidden states, plus: main_proj/main_norm (fuse target layers 40/41/42), a
rank-256 Markov head (markov_w1/markov_w2), a confidence head (conf_proj), and
the output head (hc_head_* + norm) on the last stage. embed_tokens + lm_head are
SHARED from the target (absent here).

Drafter quant policy mirrors the proven single-head MTP GGUF: routed experts
Q4_K, attn/proj/shared-experts/main_proj Q8_0, norms/hc/gate F32, markov/conf
F16. The 3 layers are byte-identical in HF format to base layers, so the FP8
(e4m3 + ue8m0 block-128) / FP4 (e2m1 packed I8 + ue8m0 block-32) dequant is the
same as deepseek4-quantize.c (ported here to numpy).

Usage:
  python dspark_extract.py --src DIR_WITH_3_SHARDS --out dspark.gguf \
        [--validate] [--experts q4_k|q8_0|q2_k] [--experts-down q4_k|q8_0|q2_k] \
        [--checkpoint-variant vision-exp --source-revision SHA --source-url URL]

The ship recipe (0731 and Vision-Exp drafters alike) is `--experts q2_k
--validate`; the three stamping flags are mandatory for a Vision-Exp
extraction (the engine refuses an unstamped drafter beside a Vision-Exp base).
"""
import argparse, json, struct, os, sys, gc
import numpy as np
import gguf
from gguf import GGMLQuantizationType as QT
import gguf.quants as gq

# ---------------------------------------------------------------- dequant tables
def _build_e4m3_table():
    t = np.zeros(256, dtype=np.float32)
    for x in range(256):
        a = x & 0x7f
        sign = -1.0 if (x & 0x80) else 1.0
        if a == 0:
            t[x] = sign * 0.0; continue
        if a == 0x7f:
            t[x] = 0.0; continue          # NaN slot -> 0 (matches C)
        exp = (x >> 3) & 0x0f
        man = x & 0x07
        if exp == 0:
            v = man * (2.0 ** -9)
        else:
            v = (1.0 + man / 8.0) * (2.0 ** (exp - 7))
        t[x] = sign * v
    return t
E4M3 = _build_e4m3_table()

FP4 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0], dtype=np.float32)

def e8m0_to_f32(e_u8):
    e = e_u8.astype(np.uint32)
    bits = np.where(e == 0, np.uint32(0x00400000), e << 23).astype(np.uint32)
    return bits.view(np.float32)

def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)

# ---------------------------------------------------------------- safetensors db
class STDB:
    def __init__(self, src):
        self.entries = {}   # name -> (path, dtype, shape, begin, end, data0)
        idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
        shards = sorted(set(idx["weight_map"].values()))
        for sh in shards:
            path = os.path.join(src, sh)
            if not os.path.exists(path):
                continue   # only the DSpark shards (46-48) are downloaded
            with open(path, "rb") as fp:
                hlen = struct.unpack("<Q", fp.read(8))[0]
                hdr = json.loads(fp.read(hlen))
            data0 = 8 + hlen
            for name, info in hdr.items():
                if name == "__metadata__":
                    continue
                b, e = info["data_offsets"]
                self.entries[name] = (path, info["dtype"], tuple(info["shape"]), b, e, data0)

    def has(self, name): return name in self.entries

    def raw(self, name):
        path, dt, shape, b, e, d0 = self.entries[name]
        cnt = e - b
        with open(path, "rb") as fp:
            fp.seek(d0 + b)
            buf = fp.read(cnt)
        return dt, shape, np.frombuffer(buf, dtype=np.uint8)

    def f32(self, name):
        """Dequantize any HF tensor to a logical-shape float32 numpy array."""
        dt, shape, raw = self.raw(name)
        if dt == "F32":
            return raw.view(np.float32).reshape(shape)
        if dt == "BF16":
            return bf16_to_f32(raw.view(np.uint16)).reshape(shape)
        if dt == "F16":
            return raw.view(np.float16).astype(np.float32).reshape(shape)
        if dt == "F8_E4M3":
            return E4M3[raw].reshape(shape)
        raise SystemExit(f"f32(): unhandled direct dtype {dt} for {name}")

    def fp8(self, name):
        """FP8 e4m3 weight + ue8m0 block-128 scale -> f32 [out,in]."""
        dt, shape, raw = self.raw(name)
        assert dt == "F8_E4M3" and len(shape) == 2, (name, dt, shape)
        out, inn = shape
        w = E4M3[raw].reshape(out, inn)
        _, sshape, sraw = self.raw(name.rsplit(".", 1)[0] + ".scale")
        assert out % 128 == 0 and inn % 128 == 0, (name, shape)
        sc = e8m0_to_f32(sraw).reshape(out // 128, inn // 128)
        sc = np.repeat(np.repeat(sc, 128, axis=0), 128, axis=1)
        return w * sc

    def fp4(self, name):
        """FP4 e2m1 packed-I8 weight + ue8m0 block-32 scale -> f32 [out,in]."""
        dt, shape, raw = self.raw(name)
        assert dt == "I8" and len(shape) == 2, (name, dt, shape)
        out, packed = shape
        inn = packed * 2
        w8 = raw.reshape(out, packed)
        lo = FP4[w8 & 0x0f]
        hi = FP4[(w8 >> 4) & 0x0f]
        wf = np.empty((out, inn), dtype=np.float32)
        wf[:, 0::2] = lo
        wf[:, 1::2] = hi
        _, sshape, sraw = self.raw(name.rsplit(".", 1)[0] + ".scale")
        nblk = inn // 32
        assert tuple(sshape) == (out, nblk), (name, sshape, (out, nblk))
        sc = e8m0_to_f32(sraw).reshape(out, nblk)
        sc = np.repeat(sc, 32, axis=1)
        return wf * sc

# ---------------------------------------------------------------- Q4_K (RTN, numpy)
# gguf-py only implements Q8_0/BF16 quantize; k-quants raise NotImplementedError.
# ds4 routed experts must be IQ2_XXS/Q2_K/Q4_K, so we hand-roll Q4_K to match the
# ggml block_q4_K layout (144 B / 256 weights: d f16, dmin f16, scales[12], qs[128]).
# Validated by round-tripping through gguf.quants.dequantize (the authoritative reader).
def quantize_q4_k(arr):
    K = arr.shape[-1]
    assert K % 256 == 0, K
    rows = np.ascontiguousarray(arr).reshape(-1, K).astype(np.float32)
    nrow, nsb = rows.shape[0], K // 256
    x = rows.reshape(nrow, nsb, 8, 32)
    mn = np.minimum(x.min(axis=3), 0.0)               # [.,.,8]  <=0
    mx = np.maximum(x.max(axis=3), 0.0)               # >=0
    d = (mx - mn) / 15.0                               # sub-scale >=0
    submin = -mn                                       # >=0
    dsafe = np.where(d > 0, d, 1.0)
    q = np.clip(np.rint((x - mn[..., None]) / dsafe[..., None]), 0, 15).astype(np.uint8)
    dscale = d.max(axis=2) / 63.0                      # super-scale for scales
    dmins = submin.max(axis=2) / 63.0
    qsc = np.clip(np.rint(d / np.where(dscale > 0, dscale, 1.0)[..., None]), 0, 63).astype(np.uint8)
    qmn = np.clip(np.rint(submin / np.where(dmins > 0, dmins, 1.0)[..., None]), 0, 63).astype(np.uint8)
    scales = np.zeros((nrow, nsb, 12), dtype=np.uint8)
    for j in range(4):
        scales[..., j]     = (qsc[..., j] & 63) | ((qsc[..., j + 4] >> 4) << 6)
        scales[..., j + 4] = (qmn[..., j] & 63) | ((qmn[..., j + 4] >> 4) << 6)
        scales[..., j + 8] = (qsc[..., j + 4] & 0x0F) | ((qmn[..., j + 4] & 0x0F) << 4)
    qs = np.zeros((nrow, nsb, 128), dtype=np.uint8)
    for c in range(4):
        lo = q[..., 2 * c, :]
        hi = q[..., 2 * c + 1, :]
        qs[..., 32 * c:32 * c + 32] = (lo & 0x0F) | ((hi & 0x0F) << 4)
    d16 = np.ascontiguousarray(dscale.astype(np.float16)).view(np.uint8).reshape(nrow, nsb, 2)
    dm16 = np.ascontiguousarray(dmins.astype(np.float16)).view(np.uint8).reshape(nrow, nsb, 2)
    block = np.concatenate([d16, dm16, scales, qs], axis=2)   # [nrow,nsb,144]
    return block.reshape(*arr.shape[:-1], nsb * 144)

# ---------------------------------------------------------------- Q2_K (RTN, numpy)
# ggml block_q2_K layout (84 B / 256 weights, NOTE d/dmin at the END, unlike Q4_K):
#   uint8 scales[16]  low nibble = 4-bit sub-block scale, high nibble = 4-bit min
#   uint8 qs[64]      2-bit quants; byte l of each 128-half packs positions
#                     h+l | h+32+l<<2 | h+64+l<<4 | h+96+l<<6   (h in {0,128})
#   f16 d, f16 dmin   super-block scales for the 4-bit scale/min indices
# Dequant contract (llama.cpp dequantize_row_q2_K): y = d*sc[g]*q - dmin*m[g],
# g = pos//16. Elements are quantized against the RECONSTRUCTED d*sc / dmin*m
# (not the pre-quantization group scales) so encode matches decoder arithmetic.
def quantize_q2_k(arr):
    K = arr.shape[-1]
    assert K % 256 == 0, K
    rows = np.ascontiguousarray(arr).reshape(-1, K).astype(np.float32)
    nrow, nsb = rows.shape[0], K // 256
    x = rows.reshape(nrow, nsb, 16, 16)                # 16 groups of 16
    mn = np.minimum(x.min(axis=3), 0.0)                # [.,.,16] <=0
    mx = np.maximum(x.max(axis=3), 0.0)                # >=0
    dl = (mx - mn) / 3.0                               # group scale >=0
    ml = -mn                                           # group min  >=0
    d = dl.max(axis=2) / 15.0                          # super scale [nrow,nsb]
    dmin = ml.max(axis=2) / 15.0
    d16 = d.astype(np.float16); dmin16 = dmin.astype(np.float16)
    dr = d16.astype(np.float32); dminr = dmin16.astype(np.float32)  # as decoder sees them
    qsc = np.clip(np.rint(dl / np.where(dr > 0, dr, 1.0)[..., None]), 0, 15).astype(np.uint8)
    qmn = np.clip(np.rint(ml / np.where(dminr > 0, dminr, 1.0)[..., None]), 0, 15).astype(np.uint8)
    dlr = dr[..., None] * qsc                          # reconstructed group scale
    mlr = dminr[..., None] * qmn                       # reconstructed group min
    dlsafe = np.where(dlr > 0, dlr, 1.0)
    q = np.clip(np.rint((x + mlr[..., None]) / dlsafe[..., None]), 0, 3).astype(np.uint8)
    scales = (qsc | (qmn << 4)).astype(np.uint8)       # [nrow,nsb,16]
    qv = q.reshape(nrow, nsb, 2, 4, 32)                # [.,.,half,shift,lane]
    qs = (qv[..., 0, :] | (qv[..., 1, :] << 2) |
          (qv[..., 2, :] << 4) | (qv[..., 3, :] << 6)).astype(np.uint8)  # [.,.,2,32]
    qs = qs.reshape(nrow, nsb, 64)
    db = np.ascontiguousarray(d16).view(np.uint8).reshape(nrow, nsb, 2)
    dmb = np.ascontiguousarray(dmin16).view(np.uint8).reshape(nrow, nsb, 2)
    block = np.concatenate([scales, qs, db, dmb], axis=2)   # [nrow,nsb,84]
    return block.reshape(*arr.shape[:-1], nsb * 84)

def quantize_any(arr, qtype):
    if qtype == QT.Q4_K:
        return quantize_q4_k(arr.astype(np.float32))
    if qtype == QT.Q2_K:
        return quantize_q2_k(arr.astype(np.float32))
    return gq.quantize(arr.astype(np.float32), qtype)

# ---------------------------------------------------------------- gguf writer
def add(writer, name, arr, qtype):
    arr = np.ascontiguousarray(arr)
    if qtype == QT.F32:
        writer.add_tensor(name, arr.astype(np.float32))
    elif qtype == QT.F16:
        writer.add_tensor(name, arr.astype(np.float16))
    else:
        # pre-quantized: gguf derives the logical shape from the byte shape, so
        # pass ONLY raw_dtype (no raw_shape).
        q = quantize_any(arr, qtype)
        writer.add_tensor(name, q, raw_dtype=qtype)
    print(f"  + {name:34s} {qtype.name:5s} {tuple(arr.shape)}", flush=True)

def add_experts(writer, name, db, layer, hf_part, qtype, n_expert=256):
    """Quantize 256 routed experts INCREMENTALLY (per-expert dequant->quantize)
    to keep peak RAM ~1 GB instead of stacking all to f32 (~9 GB). Builds the
    pre-quantized [n_expert, out, bytes_per_row] uint8 tensor gguf expects."""
    blocks = None
    for x in range(n_expert):
        e = db.fp4(f"mtp.{layer}.ffn.experts.{x}.{hf_part}")   # [out,in] f32
        qb = quantize_any(e, qtype)
        if blocks is None:
            blocks = np.empty((n_expert,) + qb.shape, dtype=np.uint8)
        blocks[x] = qb
        del e, qb
    writer.add_tensor(name, blocks, raw_dtype=qtype)
    print(f"  + {name:34s} {qtype.name:5s} experts={n_expert} {blocks.shape}", flush=True)
    del blocks

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--experts", default="q4_k", choices=["q4_k", "q8_0", "q2_k"])
    ap.add_argument("--experts-down", default=None, choices=["q4_k", "q8_0", "q2_k"],
                    help="override quant for ffn_down_exps (w2) only. The BASE model's "
                         "expert tiers are IQ2_XXS gate/up + Q2_K down; the drafter "
                         "path dequantizes any of q4_k/q8_0/q2_k, and the ship drafters "
                         "(0731 and vision-exp) use q2_k for gate, up and down")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--checkpoint-variant", default=None,
                    help="stamp deepseek4.checkpoint_variant (e.g. vision-exp); "
                         "the engine refuses a drafter whose variant differs from the base")
    ap.add_argument("--source-revision", default=None,
                    help="stamp general.source.revision (HF commit sha of the source checkpoint)")
    ap.add_argument("--source-url", default=None, help="stamp general.source.url")
    args = ap.parse_args()
    # Checkpoint identity is derived from the source, not trusted to memory:
    # a Vision-Exp config.json carries vision_* keys, and a drafter extracted
    # from it MUST be stamped (the engine refuses an unstamped drafter beside
    # a Vision-Exp base, and an unstamped one beside a 0731 base would pair
    # silently with collapsed yield).
    cfg_path = os.path.join(args.src, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        src_is_vision = any(k.startswith("vision_") for k in cfg)
        if src_is_vision and not args.checkpoint_variant:
            args.checkpoint_variant = "vision-exp"
            print("note: source config.json carries vision_* keys; stamping "
                  "deepseek4.checkpoint_variant=vision-exp", file=sys.stderr)
        if not src_is_vision and args.checkpoint_variant == "vision-exp":
            ap.error("--checkpoint-variant vision-exp but the source config.json has no vision_* keys")
    if args.checkpoint_variant == "vision-exp" and not args.source_revision:
        ap.error("Vision-Exp source: --source-revision (the HF commit sha of the checkpoint) "
                 "is required; the engine refuses a Vision-Exp drafter without it")
    QMAP = {"q4_k": QT.Q4_K, "q8_0": QT.Q8_0, "q2_k": QT.Q2_K}
    EXP = QMAP[args.experts]
    EXPD = QMAP[args.experts_down] if args.experts_down else EXP

    db = STDB(args.src)
    # use_temp_file=True streams tensor data to a temp file instead of holding the
    # whole ~12 GB GGUF in RAM (matters when co-resident with a big inference server).
    w = gguf.GGUFWriter(args.out, "deepseek4-dspark", use_temp_file=True)
    w.add_uint32("deepseek4.dspark.layer_count", args.layers)
    w.add_uint32("deepseek4.dspark.block_size", 5)
    w.add_uint32("deepseek4.dspark.markov_rank", 256)
    w.add_uint32("deepseek4.dspark.noise_token_id", 128799)
    w.add_uint32("deepseek4.dspark.expert_count", 256)
    w.add_array("deepseek4.dspark.target_layers", [40, 41, 42])
    # Checkpoint identity: Vision-Exp drafters carry the base's variant key and
    # pinned revision so the engine can refuse cross-generation pairings.
    if args.checkpoint_variant:
        w.add_string("deepseek4.checkpoint_variant", args.checkpoint_variant)
    if args.source_revision:
        w.add_string("general.source.revision", args.source_revision)
    if args.source_url:
        w.add_string("general.source.url", args.source_url)

    fp8 = lambda d, n: d.fp8(n)
    fp4 = lambda d, n: d.fp4(n)

    for L in range(args.layers):
        p = f"dspark.{L}."
        m = f"mtp.{L}."
        # attention projections (FP8 -> Q8_0)
        add(w, p+"attn_q_a.weight",     db.fp8(m+"attn.wq_a.weight"), QT.Q8_0)
        add(w, p+"attn_q_b.weight",     db.fp8(m+"attn.wq_b.weight"), QT.Q8_0)
        add(w, p+"attn_kv.weight",      db.fp8(m+"attn.wkv.weight"),  QT.Q8_0)
        add(w, p+"attn_output_a.weight",db.fp8(m+"attn.wo_a.weight"), QT.Q8_0)
        add(w, p+"attn_output_b.weight",db.fp8(m+"attn.wo_b.weight"), QT.Q8_0)
        add(w, p+"attn_q_a_norm.weight",  db.f32(m+"attn.q_norm.weight"),  QT.F32)
        add(w, p+"attn_kv_a_norm.weight", db.f32(m+"attn.kv_norm.weight"), QT.F32)
        add(w, p+"attn_sinks.weight",     db.f32(m+"attn.attn_sink"),      QT.F32)
        add(w, p+"attn_norm.weight",      db.f32(m+"attn_norm.weight"),    QT.F32)
        add(w, p+"ffn_norm.weight",       db.f32(m+"ffn_norm.weight"),     QT.F32)
        # HC mix params (F32)
        for hk in ("hc_attn_fn","hc_attn_base","hc_attn_scale",
                   "hc_ffn_fn","hc_ffn_base","hc_ffn_scale"):
            add(w, p+hk+".weight", db.f32(m+hk), QT.F32)
        # MoE gate (F32) + bias
        add(w, p+"ffn_gate_inp.weight", db.f32(m+"ffn.gate.weight"), QT.F32)
        add(w, p+"exp_probs_b.bias",    db.f32(m+"ffn.gate.bias"),   QT.F32)
        # shared experts (FP8 -> Q8_0)
        add(w, p+"ffn_gate_shexp.weight", db.fp8(m+"ffn.shared_experts.w1.weight"), QT.Q8_0)
        add(w, p+"ffn_up_shexp.weight",   db.fp8(m+"ffn.shared_experts.w3.weight"), QT.Q8_0)
        add(w, p+"ffn_down_shexp.weight", db.fp8(m+"ffn.shared_experts.w2.weight"), QT.Q8_0)
        # routed experts (FP4 -> Q4_K/Q8_0), stacked [256,out,in]
        for ds4part, hfpart, qt in (("ffn_gate_exps","w1",EXP),("ffn_up_exps","w3",EXP),("ffn_down_exps","w2",EXPD)):
            add_experts(w, p+ds4part+".weight", db, L, hfpart+".weight", qt)
            gc.collect()

    # DSpark-specific heads
    add(w, "dspark.main_proj.weight", db.fp8("mtp.0.main_proj.weight"), QT.Q8_0)
    add(w, "dspark.main_norm.weight", db.f32("mtp.0.main_norm.weight"), QT.F32)
    add(w, "dspark.markov_w1.weight", db.f32("mtp.2.markov_head.markov_w1.weight"), QT.F16)
    add(w, "dspark.markov_w2.weight", db.f32("mtp.2.markov_head.markov_w2.weight"), QT.F16)
    add(w, "dspark.conf_proj.weight", db.f32("mtp.2.confidence_head.proj.weight"), QT.F32)
    add(w, "dspark.hc_head_fn.weight",    db.f32("mtp.2.hc_head_fn"),    QT.F32)
    add(w, "dspark.hc_head_base.weight",  db.f32("mtp.2.hc_head_base"),  QT.F32)
    add(w, "dspark.hc_head_scale.weight", db.f32("mtp.2.hc_head_scale"), QT.F32)
    add(w, "dspark.norm.weight", db.f32("mtp.2.norm.weight"), QT.F32)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("wrote", args.out)

    if args.validate:
        # spot-check: re-read one fp4 expert + one fp8 attn directly, compare stats
        a = db.fp8("mtp.0.attn.wq_a.weight")
        e0 = db.fp4("mtp.0.ffn.experts.0.w1.weight")
        print(f"VALIDATE wq_a: shape {a.shape} absmax {np.abs(a).max():.4f} mean {a.mean():.5f}")
        print(f"VALIDATE exp0.w1: shape {e0.shape} absmax {np.abs(e0).max():.4f} mean {e0.mean():.5f}")

if __name__ == "__main__":
    main()
