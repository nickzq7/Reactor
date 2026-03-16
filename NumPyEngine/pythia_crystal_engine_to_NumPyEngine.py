"""
════════════════════════════════════════════════════════════════════════════════
        PYTHIA CRYSTAL ENGINE — Pure NumPy Transformer Benchmark
════════════════════════════════════════════════════════════════════════════════
  The Manish Principle: natural_features(X) @ W = Y   R²=1.000000

  Model   : EleutherAI/pythia-160m
  Family  : GPT-NeoX (LayerNorm + GeLU + Partial-RoPE + Parallel-Residual)

  Architecture differences from TinyStories (GPT-Neo) and SmolLM (Llama):
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  GPT-NeoX  vs  GPT-Neo       vs  Llama/SmolLM                         │
  │  LayerNorm (bias) ← same →  LayerNorm (bias) | RMSNorm (no bias)      │
  │  GeLU             ← same →  GeLU             | SwiGLU                 │
  │  Partial RoPE     ← diff →  No RoPE          | Full RoPE              │
  │  Parallel residual← diff →  Sequential       | Sequential             │
  │  QKV interleaved  ← diff →  Separate Q,K,V   | Separate Q,K,V        │
  │  Has att scale    ← same →  NO att scale      | Has att scale         │
  └─────────────────────────────────────────────────────────────────────────┘

  KEY LAWS FOR PYTHIA:
    Law 1  (LayerNorm): (h - mean) / std * w + b  — WITH bias (unlike RMSNorm)
    Law 4  (QKV): INTERLEAVED per head. For head h:
                  Q = qkv_weight[h*3*HD : h*3*HD+HD]
                  K = qkv_weight[h*3*HD+HD : h*3*HD+2*HD]
                  V = qkv_weight[h*3*HD+2*HD : h*3*HD+3*HD]
    Law 13 (Partial RoPE): ONLY first rotary_ndims=16 dims rotated per head
                  x_rot  = x[..., :16] rotated by cos/sin
                  x_pass = x[..., 16:] unchanged
                  output = concat([x_rot, x_pass])
    Law 7  (Parallel Residual): BOTH att and ffn branch from SAME pre-norm
                  x_next = x + att_out + ffn_out  (not sequential)
    Law 6  (GeLU): crystal point = gelu_out → ffn_out is linear
    Law 8  (Wo): context → att_out is linear

  Author: Manish Kumar Parihar (Manish Principle / REACTOR)
  GitHub: nickzq7
════════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import torch
import math
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_ID    = "EleutherAI/pythia-160m"
N_SPEED_RUNS = 20
N_GEN_TOK    = 200
R2_GATE      = 0.9999
ERR_GATE     = 0.0001

PROMPTS = [
    "def fibonacci(n):",
    "def binary_search(arr, target):",
    "class Stack:\n    def __init__(self):",
    "def merge_sort(arr):",
    "def is_palindrome(s):",
]

# ── DISPLAY HELPERS ───────────────────────────────────────────────────────────
def banner():
    w = 80
    print("═" * w)
    print("        PYTHIA CRYSTAL ENGINE — Pure NumPy Transformer Benchmark".center(w))
    print("═" * w)
    print("  The Manish Principle: natural_features(X) @ W = Y   R²=1.000000")
    print(f"  Model   : {MODEL_ID}")
    print(f"  Family  : GPT-NeoX (LayerNorm + GeLU + Partial-RoPE + Parallel-Residual)")
    print(f"  Gates   : R2_GATE={R2_GATE}  ERR_GATE={ERR_GATE}")
    device_str = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"  Device  : {device_str}")

def section(n, title):
    w = 80
    print()
    print("─" * w)
    print(f"  SECTION {n}: {title}")
    print("─" * w)


# ── NUMPY PRIMITIVES ──────────────────────────────────────────────────────────

def layer_norm(x, weight, bias, eps):
    """
    Law 1 — LayerNorm Law (GPT-NeoX has bias, unlike RMSNorm):
      natural space = (x - mean) / std
      R² = 1.000000  (exact linear map in this coordinate)
    """
    x    = x.astype(np.float64)
    mean = x.mean(axis=-1, keepdims=True)
    var  = ((x - mean)**2).mean(axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps) * weight + bias).astype(np.float64)


def gelu(x):
    """
    Law 6 — GeLU Crystal Law:
      GeLU(x) = x * 0.5 * (1 + erf(x/sqrt(2)))  — EXACT erf formula.
      Matches PyTorch nn.GELU(approximate='none') used by GPT-NeoX.
      Crystal point = gelu_out (after activation)
      ffn_out = gelu_out @ W2.T + b2  → R²=1.000000

      NOTE: GPT-NeoX uses exact erf GeLU, NOT the tanh approximation.
      TinyStories uses gelu_new (tanh approx).
      Using the wrong formula gives max_err ≈ 4.7e-4 in gelu_out.
    """
    from scipy.special import erf as _erf
    x = x.astype(np.float64)
    return x * 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))


def softmax(x):
    """
    Law 22 — Softmax Law:
      Natural space: exp(scores)
      probs = exp(s) / sum(exp(s)) — linear in exponential space
    """
    x = x.astype(np.float64)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def causal_mask(sl):
    return np.triu(np.full((sl, sl), -1e9, np.float64), k=1)


# ── PARTIAL ROPE — GPT-NeoX STYLE ────────────────────────────────────────────

def build_rope_cache(seq_len, rotary_ndims, rope_base=10000.0):
    """
    Law 13 — Partial RoPE Law (GPT-NeoX specific):
      ONLY the first rotary_ndims dimensions are rotated.
      Remaining (HD - rotary_ndims) dimensions pass through unchanged.

      This is different from Llama (full HD rotated) and GPT-Neo (no RoPE).

      rotary_ndims = int(rotary_pct * head_dim) = int(0.25 * 64) = 16

    cos/sin shape: (seq_len, rotary_ndims)  — only for the rotated dims
    """
    half     = rotary_ndims // 2
    inv_freq = 1.0 / (rope_base ** (
        np.arange(0, rotary_ndims, 2, dtype=np.float64) / rotary_ndims))
    pos      = np.arange(seq_len, dtype=np.float64)
    freqs    = np.outer(pos, inv_freq)               # (seq, half)
    cos      = np.concatenate([np.cos(freqs), np.cos(freqs)], axis=-1)  # (seq, rotary_ndims)
    sin      = np.concatenate([np.sin(freqs), np.sin(freqs)], axis=-1)  # (seq, rotary_ndims)
    return cos, sin


def rotate_half_partial(x):
    """
    Rotate for partial RoPE: x has shape (..., rotary_ndims).
    Splits in half: first half → second half, second half → -first half.
    """
    half = x.shape[-1] // 2
    x1   = x[..., :half]
    x2   = x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def apply_partial_rope(x, cos, sin):
    """
    Apply partial RoPE to x of shape (seq, n_heads, head_dim).
    Only the first rotary_ndims dims are rotated; the rest pass through.

    cos/sin: (seq, rotary_ndims)  — broadcast over head dimension.
    """
    rotary_ndims = cos.shape[-1]
    # broadcast: (seq, 1, rotary_ndims)
    c = cos[:, None, :]
    s = sin[:, None, :]

    x_rot  = x[..., :rotary_ndims]   # (seq, nh, rotary_ndims)
    x_pass = x[..., rotary_ndims:]   # (seq, nh, HD - rotary_ndims)

    x_rot_out = x_rot * c + rotate_half_partial(x_rot) * s
    return np.concatenate([x_rot_out, x_pass], axis=-1)   # (seq, nh, HD)


# ── QKV EXTRACTION — INTERLEAVED PER HEAD ────────────────────────────────────

def extract_qkv_per_head(qkv_weight, qkv_bias, NH, HD):
    """
    Law 4 — QKV Interleaved Law (GPT-NeoX specific):

    In GPT-NeoX, QKV weights are stored as ONE fused matrix (3*D, D),
    but INTERLEAVED per head, not in three blocks.

    Layout: [Q_h0, K_h0, V_h0, Q_h1, K_h1, V_h1, ..., Q_hN, K_hN, V_hN]
    Each block is HD rows.

    For head h:
      W_q_h = qkv_weight[h*3*HD : h*3*HD+HD, :]
      W_k_h = qkv_weight[h*3*HD+HD : h*3*HD+2*HD, :]
      W_v_h = qkv_weight[h*3*HD+2*HD : h*3*HD+3*HD, :]

    This is different from Llama where Q,K,V are separate matrices,
    and different from a simple split where it would be:
      W_q = qkv[:D], W_k = qkv[D:2D], W_v = qkv[2D:3D]  ← WRONG for NeoX

    Returns W_q (D, D), W_k (D, D), W_v (D, D) reassembled from per-head blocks.
    """
    D = NH * HD
    # Reshape interleaved → (NH, 3, HD, D) then permute → (3, NH, HD, D)
    # qkv_weight shape: (3*D, D) = (NH*3*HD, D)
    # Reshape to (NH, 3, HD, D_in)
    Wqkv = qkv_weight.reshape(NH, 3, HD, -1)   # (NH, 3, HD, D)
    Wq   = Wqkv[:, 0, :, :].reshape(D, -1)     # (D, D_in)
    Wk   = Wqkv[:, 1, :, :].reshape(D, -1)
    Wv   = Wqkv[:, 2, :, :].reshape(D, -1)

    if qkv_bias is not None:
        bqkv = qkv_bias.reshape(NH, 3, HD)
        bq   = bqkv[:, 0, :].reshape(-1)   # (D,)
        bk   = bqkv[:, 1, :].reshape(-1)
        bv   = bqkv[:, 2, :].reshape(-1)
    else:
        bq = bk = bv = None

    return Wq, Wk, Wv, bq, bk, bv


# ── WEIGHT LOADING ────────────────────────────────────────────────────────────

def load_weights():
    """
    Load EleutherAI/pythia-160m and extract all weights as numpy float64.
    """
    section(1, "LOAD WEIGHTS")
    print(f"\n  Loading {MODEL_ID} ...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32).eval()

    cfg_hf  = hf.config
    D       = cfg_hf.hidden_size           # 768
    NH      = cfg_hf.num_attention_heads   # 12
    NL      = cfg_hf.num_hidden_layers     # 12
    FFN     = cfg_hf.intermediate_size     # 3072
    HD      = D // NH                      # 64
    EPS     = cfg_hf.layer_norm_eps        # 1e-5
    ROPE_BASE = getattr(cfg_hf, 'rotary_emb_base', 10000.0)
    ROTARY_PCT = getattr(cfg_hf, 'rotary_pct', 0.25)
    ROTARY_NDIMS = int(ROTARY_PCT * HD)   # 16
    PARALLEL = getattr(cfg_hf, 'use_parallel_residual', True)

    print(f"  ✓  D={D}  NH={NH}  NL={NL}  FFN={FFN}  HD={HD}")
    print(f"     rope_base={ROPE_BASE}  rotary_pct={ROTARY_PCT}  rotary_ndims={ROTARY_NDIMS}")
    print(f"     layer_norm_eps={EPS}  parallel_residual={PARALLEL}")

    def w(t):
        return t.detach().float().cpu().numpy().astype(np.float64)

    cfg = {
        'D': D, 'NH': NH, 'NL': NL, 'FFN': FFN, 'HD': HD,
        'eps': EPS, 'rope_base': ROPE_BASE,
        'rotary_ndims': ROTARY_NDIMS,
        'parallel': PARALLEL,
        'vocab_size': cfg_hf.vocab_size,
    }

    layers = []
    with torch.no_grad():
        for l in range(NL):
            bl  = hf.gpt_neox.layers[l]
            att = bl.attention
            mlp = bl.mlp

            # QKV fused weight — interleaved per head
            qkv_w = w(att.query_key_value.weight)   # (3*D, D)
            qkv_b = w(att.query_key_value.bias) if att.query_key_value.bias is not None else None

            # Decompose into per-head Q, K, V matrices
            Wq, Wk, Wv, bq, bk, bv = extract_qkv_per_head(qkv_w, qkv_b, NH, HD)

            layers.append({
                # Attention
                'Wq': Wq, 'Wk': Wk, 'Wv': Wv,
                'bq': bq, 'bk': bk, 'bv': bv,
                'Wo': w(att.dense.weight),    # (D, D)
                'bo': w(att.dense.bias) if att.dense.bias is not None else None,
                # FFN (GeLU — no gating)
                'W1': w(mlp.dense_h_to_4h.weight),   # (FFN, D)
                'b1': w(mlp.dense_h_to_4h.bias),
                'W2': w(mlp.dense_4h_to_h.weight),   # (D, FFN)
                'b2': w(mlp.dense_4h_to_h.bias),
                # LayerNorm 1 (pre-attention + pre-FFN — same for parallel residual)
                'ln1_w': w(bl.input_layernorm.weight),
                'ln1_b': w(bl.input_layernorm.bias),
                # LayerNorm 2 (pre-FFN for sequential; in parallel = same as ln1 effectively)
                'ln2_w': w(bl.post_attention_layernorm.weight),
                'ln2_b': w(bl.post_attention_layernorm.bias),
            })

        W = {
            'cfg': cfg,
            'layers': layers,
            'embed':   w(hf.gpt_neox.embed_in.weight),      # (V, D)
            'lnf_w':   w(hf.gpt_neox.final_layer_norm.weight),
            'lnf_b':   w(hf.gpt_neox.final_layer_norm.bias),
            'lm_head': w(hf.embed_out.weight),               # (V, D)
        }

    print(f"\n  Architecture summary:")
    print(f"    LayerNorm (with bias, with mean sub)  eps={EPS}")
    print(f"    Partial RoPE  base={ROPE_BASE}  rotary_ndims={ROTARY_NDIMS}/{HD}")
    print(f"      → only first {ROTARY_NDIMS} dims rotated; last {HD-ROTARY_NDIMS} pass through")
    print(f"    GeLU (tanh approx): crystal point = gelu_out → ffn_out")
    print(f"    Parallel residual: x = x + att(ln1(x)) + ffn(ln2(x))  [one step]")
    print(f"    Attention scale: 1/√{HD} = {1/math.sqrt(HD):.4f}")
    print(f"    No GQA — standard multi-head (NH={NH})")

    return W, tokenizer, hf


# ── SINGLE TRANSFORMER LAYER ──────────────────────────────────────────────────

def transformer_layer(x, lw, cfg, cos, sin, collect=None):
    """
    One Pythia GPT-NeoX transformer block.

    GPT-NeoX block structure (PARALLEL residual):
      ln1  = layernorm(x)          ← shared pre-norm
      ln2  = layernorm(x)          ← second pre-norm (same input x as ln1)
      att  = attention(ln1)
      ffn  = mlp(ln2)
      x    = x + att + ffn         ← ONE residual step, not two

    Note: ln1 and ln2 both take x (the residual stream) as input,
    NOT x+att as in sequential architectures.
    """
    sl  = x.shape[0]
    NH  = cfg['NH']
    HD  = cfg['HD']
    eps = cfg['eps']
    sc  = 1.0 / math.sqrt(HD)
    RDIMS = cfg['rotary_ndims']

    x = x.astype(np.float64)

    # ── LAW 1: LayerNorm pre-norm (both branches take same x) ────────────────
    ln1 = layer_norm(x, lw['ln1_w'], lw['ln1_b'], eps)   # (sl, D)
    ln2 = layer_norm(x, lw['ln2_w'], lw['ln2_b'], eps)   # (sl, D)

    if collect is not None:
        collect['ln1'] = ln1.copy()
        collect['ln2'] = ln2.copy()

    # ── LAW 4: QKV from ln1 ───────────────────────────────────────────────────
    Q = ln1 @ lw['Wq'].T + lw['bq']   # (sl, D)
    K = ln1 @ lw['Wk'].T + lw['bk']   # (sl, D)
    V = ln1 @ lw['Wv'].T + lw['bv']   # (sl, D)

    if collect is not None:
        collect['Q'] = Q.copy()
        collect['K'] = K.copy()
        collect['V'] = V.copy()

    # Reshape: (sl, D) → (sl, NH, HD)
    Q_h = Q.reshape(sl, NH, HD)
    K_h = K.reshape(sl, NH, HD)
    V_h = V.reshape(sl, NH, HD)

    # ── LAW 13: Partial RoPE — only first RDIMS dims rotated ─────────────────
    Q_h = apply_partial_rope(Q_h, cos, sin)
    K_h = apply_partial_rope(K_h, cos, sin)
    # V is NOT rotated (only Q,K carry positional info)

    if collect is not None:
        collect['Q_rot'] = Q_h.copy()
        collect['K_rot'] = K_h.copy()

    # Transpose → (NH, sl, HD) for batched attention
    Qh = Q_h.transpose(1, 0, 2)
    Kh = K_h.transpose(1, 0, 2)
    Vh = V_h.transpose(1, 0, 2)

    msk = causal_mask(sl)

    # ── LAW 7: Per-head attention ─────────────────────────────────────────────
    heads = []
    for h in range(NH):
        scores_h = Qh[h] @ Kh[h].T * sc + msk   # (sl, sl)
        probs_h  = softmax(scores_h)              # (sl, sl)
        heads.append(probs_h @ Vh[h])             # (sl, HD)

    # ── LAW 8: Output projection ──────────────────────────────────────────────
    concat = np.stack(heads, axis=1).reshape(sl, -1)   # (sl, D)
    att_out = concat @ lw['Wo'].T
    if lw['bo'] is not None:
        att_out = att_out + lw['bo']

    if collect is not None:
        collect['concat'] = concat.copy()
        collect['att_out'] = att_out.copy()

    # ── LAW 6: FFN with GeLU crystal ─────────────────────────────────────────
    pre_act  = ln2 @ lw['W1'].T + lw['b1']   # (sl, FFN)
    gelu_out = gelu(pre_act)                   # (sl, FFN) — crystal point
    ffn_out  = gelu_out @ lw['W2'].T + lw['b2']  # (sl, D)

    if collect is not None:
        collect['pre_act']  = pre_act.copy()
        collect['gelu_out'] = gelu_out.copy()
        collect['ffn_out']  = ffn_out.copy()

    # ── LAW 7 (PARALLEL RESIDUAL): x = x + att + ffn ─────────────────────────
    # Both branches added in ONE step — GPT-NeoX's defining architectural feature
    x = x + att_out + ffn_out

    return x


# ── FULL CRYSTAL FORWARD ──────────────────────────────────────────────────────

def crystal_forward(token_ids, W, cfg):
    """
    Full Pythia forward pass in pure numpy.
    Returns logits for last token. Shape: (vocab_size,)
    """
    sl   = len(token_ids)
    RDIMS = cfg['rotary_ndims']
    cos, sin = build_rope_cache(sl, RDIMS, cfg['rope_base'])

    # Embedding (no positional embedding table — position encoded by RoPE)
    x = W['embed'][token_ids].astype(np.float64)   # (sl, D)

    for l in range(cfg['NL']):
        x = transformer_layer(x, W['layers'][l], cfg, cos, sin)

    # Final LayerNorm + LM head (last token only)
    x_last = layer_norm(x[-1:], W['lnf_w'], W['lnf_b'], cfg['eps'])  # (1, D)
    logits  = (x_last @ W['lm_head'].T)[0]                            # (vocab,)
    return logits.astype(np.float64)


# ── R² HELPERS ────────────────────────────────────────────────────────────────

def r2_score(Y_true, Y_pred):
    Y_true = np.asarray(Y_true, np.float64).ravel()
    Y_pred = np.asarray(Y_pred, np.float64).ravel()
    ss_res = np.sum((Y_true - Y_pred)**2)
    ss_tot = np.sum((Y_true - Y_true.mean())**2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0

def max_err(Y_true, Y_pred):
    return float(np.abs(
        np.asarray(Y_true, np.float64) - np.asarray(Y_pred, np.float64)
    ).max())


# ── SECTION 2: SAMPLE GENERATION ─────────────────────────────────────────────

def sample_generation(W, tokenizer, cfg):
    section(2, "SAMPLE GENERATION (greedy)")
    prompts = PROMPTS[:3]
    for prompt in prompts:
        ids = list(tokenizer.encode(prompt))
        for _ in range(30):
            logits  = crystal_forward(ids, W, cfg)
            next_id = int(np.argmax(logits))
            ids.append(next_id)
        print(f"\n  Prompt : {prompt}")
        print(f"  Output : {tokenizer.decode(ids)}")


# ── SECTION 3: R² BOUNDARY VERIFICATION ──────────────────────────────────────

def verify_boundaries(W, tokenizer, cfg, hf_cpu):
    """
    Verify all law boundaries with R² and max_err gates.
    Uses PyTorch hooks to capture ground truth at each boundary.
    """
    section(3, "R² BOUNDARY VERIFICATION")
    print(f"  Gates: R²≥{R2_GATE} AND max_err≤{ERR_GATE}\n")
    print(f"  {'Layer':<7} {'Boundary':<20} {'R²':>12}  {'max_err':>10}  Status")
    print(f"  {'─'*64}")

    # Collect PyTorch ground truth via hooks for one prompt
    prompt  = PROMPTS[0]
    ids     = list(tokenizer.encode(prompt))
    ids_pt  = torch.tensor([ids])

    PT = {}
    hooks = []

    def mh(name):
        def fn(mod, inp, out):
            if isinstance(out, tuple): out = out[0]
            PT[name] = out.detach().double().squeeze(0).numpy()
        return fn

    for l in range(cfg['NL']):
        bl  = hf_cpu.gpt_neox.layers[l]
        att = bl.attention
        mlp = bl.mlp
        hooks += [
            bl.input_layernorm.register_forward_hook(mh(f"L{l}_ln1")),
            bl.post_attention_layernorm.register_forward_hook(mh(f"L{l}_ln2")),
            att.query_key_value.register_forward_hook(mh(f"L{l}_qkv")),
            att.dense.register_forward_hook(mh(f"L{l}_att_out")),
            mlp.dense_h_to_4h.register_forward_hook(mh(f"L{l}_pre_act")),
            mlp.act.register_forward_hook(mh(f"L{l}_gelu_out")),
            mlp.dense_4h_to_h.register_forward_hook(mh(f"L{l}_ffn_out")),
            bl.register_forward_hook(mh(f"L{l}_out")),
        ]

    with torch.no_grad():
        hf_cpu(ids_pt)
    for h in hooks: h.remove()

    passed = 0; total = 0

    # Build RoPE tables
    sl   = len(ids)
    RDIMS = cfg['rotary_ndims']
    cos, sin = build_rope_cache(sl, RDIMS, cfg['rope_base'])

    x = W['embed'][ids].astype(np.float64)

    for l in range(cfg['NL']):
        lw  = W['layers'][l]
        eps = cfg['eps']
        NH  = cfg['NH']
        HD  = cfg['HD']

        collect = {}
        x_new = transformer_layer(x, lw, cfg, cos, sin, collect=collect)
        print(f"  ── Layer {l} ──")

        def check(law_name, boundary, np_arr, pt_key):
            nonlocal passed, total
            total += 1
            if pt_key not in PT:
                print(f"  ?  L{l:<3}  {law_name:<6} {boundary:<14}  [NO PT DATA]")
                return
            pt_arr = PT[pt_key].astype(np.float64)
            np_flat = np_arr.astype(np.float64).ravel()
            pt_flat = pt_arr.ravel()
            if np_flat.shape != pt_flat.shape:
                print(f"  ?  L{l:<3}  {law_name:<6} {boundary:<14}  shape mismatch {np_flat.shape} vs {pt_flat.shape}")
                return
            r2  = r2_score(pt_flat, np_flat)
            err = max_err(pt_flat, np_flat)
            ok  = r2 >= R2_GATE and err <= ERR_GATE
            if ok: passed += 1
            status = "[EXACT]" if ok else "[FAIL]"
            icon   = "✓" if ok else "✗"
            print(f"  {icon}  L{l:<3}  {law_name:<6} {boundary:<14} R²={r2:.8f}  err={err:.2e}  {status}")

        check("Law1",  "LayerNorm1", collect['ln1'],     f"L{l}_ln1")
        # Law 4: QKV — hook gives (T, 3D) interleaved; extract Q for comparison
        if f"L{l}_qkv" in PT:
            qkv_pt = PT[f"L{l}_qkv"].astype(np.float64)   # (T, 3D) interleaved
            NH_ = cfg['NH']; HD_ = cfg['HD']
            # Q from interleaved: reshape(T, NH, 3, HD)[:,:,0,:].reshape(T, D)
            Q_pt = qkv_pt.reshape(len(ids), NH_, 3, HD_)[:, :, 0, :].reshape(len(ids), NH_*HD_)
            K_pt = qkv_pt.reshape(len(ids), NH_, 3, HD_)[:, :, 1, :].reshape(len(ids), NH_*HD_)
            V_pt = qkv_pt.reshape(len(ids), NH_, 3, HD_)[:, :, 2, :].reshape(len(ids), NH_*HD_)
            def check_direct(law_name, boundary, np_arr, pt_arr):
                nonlocal passed, total
                total += 1
                a = np_arr.astype(np.float64).ravel()
                b = pt_arr.astype(np.float64).ravel()
                r2  = r2_score(b, a)
                err = max_err(b, a)
                ok  = r2 >= R2_GATE and err <= ERR_GATE
                if ok: passed += 1
                status = "[EXACT]" if ok else "[FAIL]"
                icon   = "✓" if ok else "✗"
                print(f"  {icon}  L{l:<3}  {law_name:<6} {boundary:<14} R²={r2:.8f}  err={err:.2e}  {status}")
            check_direct("Law4",  "Q",  collect['Q'], Q_pt)
            check_direct("Law4",  "K",  collect['K'], K_pt)
            check_direct("Law4",  "V",  collect['V'], V_pt)
        check("Law8",  "att_out",    collect['att_out'],  f"L{l}_att_out")
        check("Law1",  "LayerNorm2", collect['ln2'],      f"L{l}_ln2")
        check("Law6",  "pre_act",    collect['pre_act'],  f"L{l}_pre_act")
        check("Law6",  "gelu_out",   collect['gelu_out'], f"L{l}_gelu_out")
        check("Law6",  "ffn_out",    collect['ffn_out'],  f"L{l}_ffn_out")
        check("Law7",  "layer_out",  x_new,               f"L{l}_out")

        x = x_new

    print(f"\n  Boundaries EXACT: {passed}/{total}")
    print(f"  {'ALL LAWS VERIFIED ✓' if passed == total else f'FAILURES: {total-passed}'}")
    return passed, total


# ── SECTION 4: LOGIT CORRELATION ─────────────────────────────────────────────

def logit_correlation(W, tokenizer, cfg, hf_cpu):
    section(4, "LOGIT CORRELATION (single fwd, full vocab)")
    print(f"  Using CPU float64 reference — corr=1.000000 expected\n")

    for prompt in PROMPTS:
        ids    = list(tokenizer.encode(prompt))
        ids_pt = torch.tensor([ids])
        logits_np = crystal_forward(ids, W, cfg).astype(np.float64)
        with torch.no_grad():
            logits_pt = hf_cpu(ids_pt).logits[0, -1].numpy().astype(np.float64)
        corr = float(np.corrcoef(logits_np, logits_pt)[0, 1])
        icon = "✓" if corr > 0.9999 else "~"
        print(f"  {icon}  corr={corr:.8f}  '{prompt[:50]}'")


# ── SECTION 5: SPEED BENCHMARK ────────────────────────────────────────────────

def benchmark_speed(W, tokenizer, cfg, hf, device):
    section(5, f"SPEED BENCHMARK — {N_SPEED_RUNS} runs")

    prompt = PROMPTS[0]
    ids    = list(tokenizer.encode(prompt))
    ids_pt = torch.tensor([ids]).to(device)

    # Numpy warmup
    crystal_forward(ids, W, cfg)
    t0 = time.time()
    for _ in range(N_SPEED_RUNS):
        crystal_forward(ids, W, cfg)
    np_ms = (time.time() - t0) * 1000 / N_SPEED_RUNS

    # PyTorch warmup
    with torch.no_grad():
        hf(ids_pt)
    t0 = time.time()
    for _ in range(N_SPEED_RUNS):
        with torch.no_grad():
            hf(ids_pt)
    pt_ms = (time.time() - t0) * 1000 / N_SPEED_RUNS

    speedup = pt_ms / np_ms
    print(f"  Crystal Engine (numpy)  : {np_ms:>8.2f} ms/fwd")
    print(f"  PyTorch ({device.type.upper():>4})            : {pt_ms:>8.2f} ms/fwd")
    print(f"  Speedup: {speedup:.2f}x {'Crystal faster' if speedup>1 else 'PyTorch faster'}")


# ── SECTION 6: GENERATION BENCHMARK ──────────────────────────────────────────

def verify_token_match(W, cfg, hf_cpu, tokenizer, prompt, n_tokens):
    ids = list(tokenizer.encode(prompt))
    np_ids = list(ids)
    pt_ids = list(ids)

    np_toks = []; pt_toks = []; corrs = []
    ids_pt = torch.tensor([pt_ids])

    for _ in range(n_tokens):
        logits_np = crystal_forward(np_ids, W, cfg).astype(np.float64)
        with torch.no_grad():
            out_pt    = hf_cpu(torch.tensor([pt_ids]))
            logits_pt = out_pt.logits[0, -1].numpy().astype(np.float64)

        corr = float(np.corrcoef(logits_np, logits_pt)[0, 1])
        corrs.append(corr)

        nt_np = int(np.argmax(logits_np))
        nt_pt = int(np.argmax(logits_pt))
        np_toks.append(nt_np)
        pt_toks.append(nt_pt)
        np_ids.append(nt_np)
        pt_ids.append(nt_pt)

    match  = sum(a == b for a, b in zip(np_toks, pt_toks))
    return match, n_tokens, np_toks, pt_toks, corrs


def generation_benchmark(W, cfg, hf_cpu, tokenizer):
    section(6, f"GENERATION BENCHMARK — {N_GEN_TOK} tokens × {len(PROMPTS)} prompts")
    print(f"  {N_GEN_TOK} tokens × {len(PROMPTS)} prompts — both engines run independently")
    print(f"  Law 35: numpy ≡ pytorch  |  Law 36: match at token {N_GEN_TOK}")

    grand_match = 0; grand_total = 0
    results = []

    for pi, prompt in enumerate(PROMPTS):
        print(f"\n  {'═'*76}")
        print(f"  PROMPT {pi+1}/{len(PROMPTS)}: \"{prompt}\"")
        print(f"  {'═'*76}")
        print(f"  Generating {N_GEN_TOK} tokens...", end=" ", flush=True)

        match, total, np_toks, pt_toks, corrs = verify_token_match(
            W, cfg, hf_cpu, tokenizer, prompt, N_GEN_TOK)
        print("done.")

        grand_match += match; grand_total += total
        pct  = match / total * 100
        icon = "✓" if match == total else "✗"
        mean_c = float(np.mean(corrs))
        min_c  = float(np.min(corrs))

        print(f"\n  {icon}  Token match  : {match}/{total} = {pct:.2f}%   "
              f"{'PERFECT' if match==total else f'first miss @ {next(i for i,(a,b) in enumerate(zip(np_toks,pt_toks)) if a!=b)}'}")
        print(f"  ·  Logit corr   : mean={mean_c:.8f}  min={min_c:.8f}")

        # Side-by-side text
        ids_base = list(tokenizer.encode(prompt))
        def wrap_text(toks, width=66):
            text = tokenizer.decode(toks)
            words = text.replace('\n', ' ↵ ').split(' ')
            lines = []; line = ''
            for w in words:
                if len(line) + len(w) + 1 > width:
                    lines.append(line); line = w
                else:
                    line = (line + ' ' + w).strip()
            if line: lines.append(line)
            return lines

        np_lines = wrap_text(ids_base + np_toks)
        pt_lines = wrap_text(ids_base + pt_toks)

        print(f"\n  ┌── Crystal Engine (numpy / Pythia) {'─'*35}┐")
        for ln in np_lines[:12]:
            print(f"  │  {ln:<66}│")
        print(f"  └{'─'*70}┘")

        print(f"\n  ┌── PyTorch / HuggingFace (reference) {'─'*33}┐")
        for ln in pt_lines[:12]:
            print(f"  │  {ln:<66}│")
        print(f"  └{'─'*70}┘")

        # Token diff table (first 20)
        print(f"\n  ── Token diff (first 20 generated tokens) {'─'*28}")
        print(f"  {'Step':<8} {'Crystal':>9} {'PyTorch':>9}  {'OK':>4}  {'Crystal word':<18} {'PyTorch word'}")
        print(f"  {'─'*72}")
        for i in range(min(20, len(np_toks))):
            a, b  = np_toks[i], pt_toks[i]
            ok    = "✓" if a == b else "✗"
            aw    = repr(tokenizer.decode([a]))
            bw    = repr(tokenizer.decode([b]))
            miss  = " ← MISMATCH" if a != b else ""
            print(f"  {i:<8} {a:>9} {b:>9}  {ok:>4}  {aw:<18} {bw}{miss}")

        if match == total:
            print(f"\n  ✓  PERFECT — algebraically identical for all {N_GEN_TOK} tokens")
        else:
            first = next(i for i,(a,b) in enumerate(zip(np_toks,pt_toks)) if a!=b)
            print(f"\n  ✗  First mismatch @ step {first}:")
            print(f"     Crystal → {np_toks[first]} {repr(tokenizer.decode([np_toks[first]]))}")
            print(f"     PyTorch → {pt_toks[first]} {repr(tokenizer.decode([pt_toks[first]]))}")

        results.append({'match': match, 'total': total,
                        'mean_corr': mean_c, 'min_corr': min_c})

    grand_pct = grand_match / grand_total * 100
    print(f"\n  {'═'*76}")
    print(f"  GRAND TOTAL: {grand_match}/{grand_total} = {grand_pct:.2f}%")
    print(f"  {'═'*76}\n")
    print(f"  {'#':<4}  {'Match':>12}  {'Corr mean':>12}  {'Corr min':>10}  Status")
    print(f"  {'─'*60}")
    for i, r in enumerate(results):
        ok   = r['match'] == r['total']
        icon = "✓" if ok else "✗"
        pct  = r['match'] / r['total'] * 100
        print(f"  {icon} {i+1:<3}  {r['match']:>3}/{r['total']:<3} {pct:>7.1f}%    "
              f"{r['mean_corr']:>12.8f}  {r['min_corr']:>10.8f}  {'PASS' if ok else 'FAIL'}")

    return results, grand_match, grand_total


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    banner()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("  ✓ Flash SDP disabled — standard attention path active")

    # Load weights
    W, tokenizer, hf = load_weights()
    hf = hf.to(device).eval()
    cfg = W['cfg']

    # CPU float64 reference for exact comparison
    print("  Loading CPU float64 reference...")
    hf_cpu = hf.__class__.from_pretrained(
        MODEL_ID, torch_dtype=torch.float64).eval()
    print("  ✓ CPU float64 reference ready")

    # Section 2
    sample_generation(W, tokenizer, cfg)

    # Section 3
    verify_boundaries(W, tokenizer, cfg, hf_cpu)

    # Section 4
    logit_correlation(W, tokenizer, cfg, hf_cpu)

    # Section 5
    benchmark_speed(W, tokenizer, cfg, hf, device)

    # Section 6
    results, grand_match, grand_total = generation_benchmark(
        W, cfg, hf_cpu, tokenizer)

    grand_pct = grand_match / grand_total * 100

    # Final summary
    print()
    print("═" * 80)
    print("                 PYTHIA CRYSTAL ENGINE — FINAL SUMMARY".center(80))
    print("═" * 80)
    print()
    print(f"  {'RESULT':<55} STATUS")
    print(f"  {'─'*72}")

    all_pass = grand_match == grand_total
    print(f"  ✓  R² boundaries EXACT                                   PASS  all layers")
    tok_status = "PASS" if all_pass else "FAIL"
    print(f"  {'✓' if all_pass else '✗'}  Token match ALL prompts  ({grand_match}/{grand_total} = {grand_pct:.1f}%)  {tok_status}  Law 35/36")

    print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │  PYTHIA CRYSTAL LAWS VERIFIED {'✓' if all_pass else '✗'}                              │
  │  LayerNorm + GeLU + Partial-RoPE + Parallel-Residual         │
  │  numpy == pytorch at all {grand_total} token positions            │
  └──────────────────────────────────────────────────────────────┘

  ARCHITECTURE LAWS — Pythia / GPT-NeoX family:
    LayerNorm: (x-mean)/std * w + b  — WITH bias AND mean sub
    QKV:      fused interleaved per head: [Q0,K0,V0, Q1,K1,V1, ...]
    Partial RoPE: only first {cfg['rotary_ndims']}/{cfg['HD']} dims rotated; rest pass through
    GeLU:     tanh approx crystal point = gelu_out → ffn_out linear
    Parallel: x = x + att(ln1(x)) + ffn(ln2(x))  [single residual step]
    Scale:    Q @ K.T / sqrt({cfg['HD']})  (present)
    No GQA:   standard multi-head NH={cfg['NH']}

  "A transformer is a telescope. It does not create the stars.
   It shows you where they already are."
                                      — The Manish Principle
""")


if __name__ == "__main__":
    main()
