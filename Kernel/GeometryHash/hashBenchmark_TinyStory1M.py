"""
GEOMETRY HASH — Compact Fingerprint of Transformer Weights
===========================================================
Like SHA but encodes GEOMETRY not content.

Each matrix gets a 6-char geometry code:
  [S][RR][EE][CC]
  S  = spectrum: P=PEAKED  M=MODERATE  X=MIXED  F=FLAT
  RR = eff_rank as 2 hex digits (00-FF)
  EE = cumE@10  scaled 0-255 → 2 hex digits
  CC = log10(cond) scaled 0-15 → 1 hex digit + sign flag

Full model fingerprint = all layer codes concatenated with separators.

Two identical models → identical geometry hash.
Two different architectures → different geometry hash.
Same model, different training → different geometry hash.

Run: python geometry_hash.py
"""

import math, warnings
import numpy as np
import torch
warnings.filterwarnings('ignore')
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.utils.import_utils as _iu
_iu.check_torch_load_is_safe = lambda: None

MODEL_ID = "roneneldan/TinyStories-1M"

# ═══════════════════════════════════════════════════════
# GEOMETRY HASH CORE
# ═══════════════════════════════════════════════════════

def matrix_geometry(M):
    """Compute raw geometry properties of matrix M."""
    M64      = M.astype(np.float64)
    _, S, _  = np.linalg.svd(M64, full_matrices=False)
    S_norm   = S / (S.sum() + 1e-15)
    entropy  = float(-np.sum(S_norm * np.log(S_norm + 1e-15)))
    eff_rank = float(np.exp(entropy))
    cond     = float(S[0] / (S[-1] + 1e-15))
    frob     = float(np.linalg.norm(M64, 'fro'))
    cumE     = np.cumsum(S**2) / (np.sum(S**2) + 1e-15)
    e10      = float(cumE[min(9,  len(cumE)-1)])
    e5       = float(cumE[min(4,  len(cumE)-1)])
    e50      = float(cumE[min(49, len(cumE)-1)])
    rank     = int(np.sum(S > 1e-3))
    sv_top4  = S[:4].tolist()
    return {
        'S': S, 'eff_rank': eff_rank, 'cond': cond,
        'frob': frob, 'rank': rank,
        'e5': e5, 'e10': e10, 'e50': e50,
        'sv_top4': sv_top4,
        'shape': list(M.shape),
    }


def encode_matrix(g):
    """
    Encode geometry dict into a 6-char code.

    Format: [S][RR][EE][C]
      S  : spectrum letter  (P M X F)
      RR : eff_rank in hex  (00–FF, capped at 255)
      EE : cumE@10 * 255    (00–FF)
      C  : log10(cond) hex  (0–F, capped at 15)

    Examples:
      P2B6C3  →  PEAKED, eff_rank=43, cumE@10=0.42, log10(cond)=3
      F3C124  →  FLAT, eff_rank=60,  cumE@10=0.07, log10(cond)=4
    """
    e10 = g['e10']
    if   e10 > 0.60: s = 'P'
    elif e10 > 0.40: s = 'M'
    elif e10 > 0.20: s = 'X'
    else:            s = 'F'

    rr = min(int(round(g['eff_rank'])), 255)
    ee = min(int(round(e10 * 255)),    255)
    c  = min(int(math.log10(max(g['cond'], 1.0))), 15)

    return f"{s}{rr:02X}{ee:02X}{c:X}"


def decode_matrix(code):
    """Decode a 6-char geometry code back to human-readable."""
    s_map = {'P': 'PEAKED', 'M': 'MODERATE', 'X': 'MIXED', 'F': 'FLAT'}
    s   = code[0]
    rr  = int(code[1:3], 16)
    ee  = int(code[3:5], 16)
    c   = int(code[5], 16)
    return {
        'spectrum'  : s_map.get(s, '?'),
        'eff_rank'  : rr,
        'cumE10'    : round(ee / 255, 3),
        'log10_cond': c,
    }


def geometry_hash(weights):
    """
    Compute the full geometry hash of a weight dict.

    Returns:
      fingerprint  : compact multi-line string (human readable)
      oneliner     : single-line hash string
      codes        : dict of name → 6-char code
    """
    cfg = weights['cfg']
    NL  = cfg['NL']
    codes = {}

    # Global matrices
    codes['WTE'] = encode_matrix(matrix_geometry(weights['wte']))
    codes['WPE'] = encode_matrix(matrix_geometry(weights['wpe']))

    # Per-layer matrices
    for l in range(NL):
        lw = weights['layers'][l]
        for name in ['Wq', 'Wk', 'Wv', 'Wo', 'W1', 'W2']:
            codes[f'L{l}.{name}'] = encode_matrix(matrix_geometry(lw[name]))
        # Effective FFN kernel
        W1W2 = lw['W2'] @ lw['W1']
        codes[f'L{l}.FFN'] = encode_matrix(matrix_geometry(W1W2))

    # Build oneliner: WTE.WPE|L0codes|L1codes|...|LNcodes
    parts = [f"{codes['WTE']}.{codes['WPE']}"]
    for l in range(NL):
        layer_part = '.'.join(codes[f'L{l}.{n}'] for n in ['Wq','Wk','Wv','Wo','W1','W2','FFN'])
        parts.append(layer_part)
    oneliner = '|'.join(parts)

    # Build fingerprint block
    lines = []
    lines.append(f"  GEOMETRY HASH — {MODEL_ID}")
    lines.append(f"  {'─'*62}")
    lines.append(f"  Format: [S=spectrum][RR=eff_rank hex][EE=cumE@10 hex][C=log10_cond hex]")
    lines.append(f"  S: P=PEAKED  M=MODERATE  X=MIXED  F=FLAT")
    lines.append(f"  {'─'*62}")
    lines.append(f"")
    lines.append(f"  {'Matrix':<14} {'Code':>8}  {'Spectrum':<10} {'eff_r':>6} {'cumE@10':>8} {'log10c':>7}")
    lines.append(f"  {'─'*62}")

    def row(name, code):
        d = decode_matrix(code)
        return (f"  {name:<14} {code:>8}  {d['spectrum']:<10}"
                f" {d['eff_rank']:>6}  {d['cumE10']:>8.3f} {d['log10_cond']:>7}")

    lines.append(row('WTE', codes['WTE']))
    lines.append(row('WPE', codes['WPE']))
    lines.append(f"  {'─'*62}")

    for l in range(NL):
        for name in ['Wq', 'Wk', 'Wv', 'Wo', 'W1', 'W2', 'FFN']:
            key = f'L{l}.{name}'
            lines.append(row(key, codes[key]))
        if l < NL - 1:
            lines.append(f"  {'·'*62}")

    lines.append(f"  {'─'*62}")
    lines.append(f"")
    lines.append(f"  ONE-LINE FINGERPRINT:")
    # Split oneliner into 64-char rows for readability
    ol = oneliner
    lines.append(f"  {ol[:64]}")
    if len(ol) > 64:
        lines.append(f"  {ol[64:128]}")
    if len(ol) > 128:
        lines.append(f"  {ol[128:]}")
    lines.append(f"")
    lines.append(f"  LAYER SUMMARY (Wq|Wk|Wv|Wo|W1|W2):")
    for l in range(NL):
        row_codes = '  '.join(codes[f'L{l}.{n}'] for n in ['Wq','Wk','Wv','Wo','W1','W2'])
        lines.append(f"  L{l}: {row_codes}")

    fingerprint = '\n'.join(lines)
    return fingerprint, oneliner, codes


def compare_hashes(codes_A, codes_B):
    """
    Compare two geometry hash dicts.
    Returns summary of matching and differing matrices.
    """
    all_keys = sorted(set(codes_A) | set(codes_B))
    matches  = []
    diffs    = []
    for k in all_keys:
        if codes_A.get(k) == codes_B.get(k):
            matches.append(k)
        else:
            diffs.append((k, codes_A.get(k, '------'), codes_B.get(k, '------')))
    return matches, diffs


# ═══════════════════════════════════════════════════════
# WEIGHT EXTRACTION
# ═══════════════════════════════════════════════════════

def extract_weights(hf_model):
    cfg = hf_model.config
    def np_(t): return t.detach().float().cpu().numpy().astype(np.float32)
    W = {}
    with torch.no_grad():
        for l in range(cfg.num_layers):
            blk = hf_model.transformer.h[l]
            W[l] = {
                'Wq': np_(blk.attn.attention.q_proj.weight),
                'Wk': np_(blk.attn.attention.k_proj.weight),
                'Wv': np_(blk.attn.attention.v_proj.weight),
                'Wo': np_(blk.attn.attention.out_proj.weight),
                'bo': np_(blk.attn.attention.out_proj.bias),
                'W1': np_(blk.mlp.c_fc.weight),
                'b1': np_(blk.mlp.c_fc.bias),
                'W2': np_(blk.mlp.c_proj.weight),
                'b2': np_(blk.mlp.c_proj.bias),
                'ln1_w': np_(blk.ln_1.weight), 'ln1_b': np_(blk.ln_1.bias),
                'ln2_w': np_(blk.ln_2.weight), 'ln2_b': np_(blk.ln_2.bias),
                'type': cfg.attention_layers[l] if l < len(cfg.attention_layers) else 'global',
            }
        return {
            'layers': W,
            'wte': np_(hf_model.transformer.wte.weight),
            'wpe': np_(hf_model.transformer.wpe.weight),
            'lnf_w': np_(hf_model.transformer.ln_f.weight),
            'lnf_b': np_(hf_model.transformer.ln_f.bias),
            'lm_head': np_(hf_model.lm_head.weight),
            'cfg': {
                'D': cfg.hidden_size, 'NH': cfg.num_attention_heads,
                'HD': cfg.hidden_size // cfg.num_attention_heads,
                'NL': cfg.num_layers, 'V': cfg.vocab_size,
                'F': hf_model.transformer.h[0].mlp.c_fc.weight.shape[0],
                'win': getattr(cfg, 'window_size', 256),
            }
        }


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

print("\n" + "═"*66)
print("  GEOMETRY HASH — Compact Fingerprint of TinyStories-1M")
print("═"*66)

# Load
print(f"\n  Loading {MODEL_ID}...")
try:
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision="refs/pr/8",
        dtype=torch.float32, use_safetensors=True).eval()
except Exception:
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32).eval()

weights = extract_weights(hf)
cfg     = weights['cfg']
print(f"  D={cfg['D']}  NH={cfg['NH']}  NL={cfg['NL']}  F={cfg['F']}")

# Compute and print geometry hash
print("\n  Computing geometry hash...")
fp, oneliner, codes = geometry_hash(weights)

print("\n" + fp)

# ── VERIFY: same model → same hash ──────────────────────
print("\n" + "═"*66)
print("  VERIFY: same model hashed twice → identical codes")
print("═"*66)
_, _, codes2 = geometry_hash(weights)
m, d = compare_hashes(codes, codes2)
print(f"\n  Same model, same hash: {len(m)}/{len(m)+len(d)} match")
if len(d) == 0:
    print("  ✓ IDENTICAL — geometry hash is deterministic")
else:
    print(f"  ✗ {len(d)} diffs (should not happen)")

# ── VERIFY: perturbed model → different hash ─────────────
print("\n" + "═"*66)
print("  VERIFY: perturbed weights → different hash")
print("═"*66)
import copy
weights_perturbed = copy.deepcopy(weights)
# Add small noise to layer 0 Wq
np.random.seed(42)
weights_perturbed['layers'][0]['Wq'] += np.random.randn(
    *weights_perturbed['layers'][0]['Wq'].shape).astype(np.float32) * 0.1

_, _, codes_p = geometry_hash(weights_perturbed)
m2, d2 = compare_hashes(codes, codes_p)
print(f"\n  Perturbed (noise on L0.Wq): {len(d2)} matrices changed")
if d2:
    print(f"  Changed matrices:")
    for name, orig, pert in d2[:5]:
        do = decode_matrix(orig); dp = decode_matrix(pert)
        print(f"    {name:<14}  orig={orig}  pert={pert}"
              f"  ({do['spectrum']}→{dp['spectrum']}"
              f"  cumE: {do['cumE10']:.3f}→{dp['cumE10']:.3f})")

# ── LEGEND ────────────────────────────────────────────────
print("\n" + "═"*66)
print("  HOW TO READ THE GEOMETRY HASH")
print("═"*66)
print(f"""
  Code format:  [S][RR][EE][C]   total 6 chars per matrix

  S  — Spectrum class (1 char)
       P = PEAKED   cumE@10 > 0.60  → low-rank, compressible
       M = MODERATE cumE@10 > 0.40  → light compression possible
       X = MIXED    cumE@10 > 0.20  → compress with caution
       F = FLAT     cumE@10 ≤ 0.20  → full-rank, do not compress

  RR — Effective rank in hex (2 chars, 00–FF)
       How many dimensions the matrix ACTUALLY uses.
       RR=10 → eff_rank=16, mostly 16 dims matter
       RR=3C → eff_rank=60, spread across 60 dims

  EE — Cumulative energy @ top-10 SVs (2 chars, 00–FF)
       EE=FF → 100% energy in top-10 → maximum peaked
       EE=80 → 50% energy in top-10  → moderate
       EE=33 → 20% energy in top-10  → flat

  C  — log10(condition number) in hex (1 char, 0–F)
       C=0 → cond≈1      (perfectly stable)
       C=3 → cond≈1000   (moderately ill-conditioned)
       C=F → cond≥1e15   (nearly singular)

  EXAMPLE:  P2B6C3
    P  → PEAKED spectrum  (concentrated energy)
    2B → eff_rank = 43    (43 of 64 dims used)
    6C → cumE@10 = 0.424  (42% energy in top-10)
    3  → log10(cond) = 3  → cond ≈ 1000

  ONE-LINE FINGERPRINT structure:
    WTE.WPE | L0codes | L1codes | ... | L7codes
    Each Lx = Wq.Wk.Wv.Wo.W1.W2.FFN  (7 codes per layer)
""")
print("═"*66)
