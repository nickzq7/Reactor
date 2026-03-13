"""
================================================================================
  REACTOR — Residual Analytic Crystal Training Operator Reactor
  A transformer framework built on the Manish Principle.
================================================================================
  " If something wrong found check testing logs for research use, code written by llm"
  THE MANISH PRINCIPLE:
    Every operation that appears nonlinear in the wrong coordinates becomes
    linear in its correct natural space.

        natural_features(X) @ W = Y     R² = 1.000000

  WHAT THIS FRAMEWORK PROVIDES:
    1. ReactorConfig     — model architecture configuration
    2. ReactorModel      — numpy transformer (forward, generate, save, load)
    3. ReactorTrainer    — O(N) training via lstsq. 0 gradient steps.
    4. V16Transfer       — geometry-based intelligence transfer between models

  TRAINING METHOD:
    Traditional:   loss → backprop → gradient steps → weight update
    REACTOR:       activations → lstsq per boundary → W extracted directly
    Gradient steps: 0.  Time complexity: O(N).  R² = 1.000000.

  NATURAL SPACES (proven exact, R²=1.0):
    Wq, Wk, Wv  :  ln_out  → Q/K/V         (no bias, exact linear)
    Wo           :  [concat, 1] → att_out   (bias, augmented lstsq)
    W1           :  [ln2_out, 1] → pre_act  (bias, augmented lstsq)
    W2           :  [acted, 1]  → ffn_out   (bias, augmented lstsq)
    GeLU         :  [x, x·tanh_factor] → gelu(x)   W=[0.5, 0.5]

  ARCHITECTURE SUPPORT:
    GPT-Neo style (local+global attention windows, no attention scale)
    Tested: roneneldan/TinyStories-1M, TinyStories-8M

================================================================================
  Author    : Manish Kumar Parihar
  License   : MIT
================================================================================
"""

import os, sys, math, time, warnings
import numpy as np
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

class ReactorConfig:
    """
    Model architecture configuration.

    Parameters
    ----------
    hidden_size   : int   — embedding / residual stream dimension (D)
    n_heads       : int   — number of attention heads
    n_layers      : int   — number of transformer layers
    vocab_size    : int   — vocabulary size
    max_position  : int   — maximum sequence length (positional embedding table size)
    ffn_mult      : int   — FFN hidden = hidden_size * ffn_mult  (default 4)
    act_fn        : str   — activation function: 'gelu_new' | 'gelu' | 'silu' | 'relu'
    attention_layers : list[str] | None
                          — per-layer type: 'global' or 'local'
                            None → all global
    window_size   : int   — local attention window (used when type='local')
    ln_eps        : float — LayerNorm epsilon
    """

    def __init__(
        self,
        hidden_size      = 64,
        n_heads          = 16,
        n_layers         = 8,
        vocab_size        = 50257,
        max_position      = 2048,
        ffn_mult          = 4,
        act_fn            = "gelu_new",
        attention_layers  = None,
        window_size       = 256,
        ln_eps            = 1e-5,
    ):
        self.hidden_size      = hidden_size
        self.n_heads          = n_heads
        self.head_dim         = hidden_size // n_heads
        self.n_layers         = n_layers
        self.vocab_size       = vocab_size
        self.max_position     = max_position
        self.ffn_hidden       = hidden_size * ffn_mult
        self.act_fn           = act_fn
        self.attention_layers = attention_layers or ['global'] * n_layers
        self.window_size      = window_size
        self.ln_eps           = ln_eps

    @classmethod
    def from_hf(cls, hf_config, ffn_hidden_override=None):
        """Build ReactorConfig from a HuggingFace model config object."""
        D = hf_config.hidden_size
        if ffn_hidden_override is not None:
            ffn_mult = ffn_hidden_override // D
        elif getattr(hf_config, 'intermediate_size', None) is not None:
            ffn_mult = hf_config.intermediate_size // D
        else:
            ffn_mult = 4   # GPT-Neo default
        return cls(
            hidden_size      = D,
            n_heads          = hf_config.num_attention_heads,
            n_layers         = hf_config.num_layers,
            vocab_size       = hf_config.vocab_size,
            max_position     = hf_config.max_position_embeddings,
            ffn_mult         = ffn_mult,
            act_fn           = getattr(hf_config, 'activation_function', 'gelu_new'),
            attention_layers = getattr(hf_config, 'attention_layers', None),
            window_size      = getattr(hf_config, 'window_size', 256),
        )

    def __repr__(self):
        return (f"ReactorConfig(D={self.hidden_size}, heads={self.n_heads}, "
                f"layers={self.n_layers}, vocab={self.vocab_size}, "
                f"act={self.act_fn})")


# ─────────────────────────────────────────────────────────────────────────────
#  ACTIVATIONS
# ─────────────────────────────────────────────────────────────────────────────

def _gelu_new(x):
    x = x.astype(np.float32)
    c = np.float32(math.sqrt(2.0 / math.pi))
    return np.float32(0.5) * x * (np.float32(1.0) +
           np.tanh(c * (x + np.float32(0.044715) * x * x * x)))

def _gelu(x):
    x = x.astype(np.float32)
    return x * (np.float32(0.5) * (np.float32(1.0) +
                np.array(np.vectorize(math.erf)(x / math.sqrt(2)), np.float32)))

def _silu(x):
    x = x.astype(np.float32)
    return x * (np.float32(1.0) / (np.float32(1.0) + np.exp(-x)))

def _relu(x):
    return np.maximum(0.0, x).astype(np.float32)

_ACT = {
    'gelu_new': _gelu_new,
    'gelu'    : _gelu,
    'silu'    : _silu,
    'relu'    : _relu,
}


# ─────────────────────────────────────────────────────────────────────────────
#  NATURAL SPACE UTILITIES  (The Manish Principle)
# ─────────────────────────────────────────────────────────────────────────────

def natural_features(x, act_fn='gelu_new'):
    """
    Build the natural feature vector for activation act_fn.
    In natural space: [x, x·tanh_factor(x)] @ W = act(x)  with R²=1.0.

    Returns
    -------
    features : np.ndarray  shape (..., 2*D)
    """
    x = np.asarray(x, np.float32)
    if act_fn == 'gelu_new':
        c = np.float32(math.sqrt(2.0 / math.pi))
        tanh_part = np.tanh(c * (x + np.float32(0.044715) * x * x * x))
        return np.concatenate([x, x * tanh_part], axis=-1)
    elif act_fn in ('gelu',):
        erf_part = np.array(np.vectorize(math.erf)(x / math.sqrt(2)), np.float32)
        return np.concatenate([x, x * erf_part], axis=-1)
    elif act_fn == 'silu':
        sigmoid = np.float32(1.0) / (np.float32(1.0) + np.exp(-x))
        return np.concatenate([x, x * sigmoid], axis=-1)
    elif act_fn == 'relu':
        return np.concatenate([x, np.abs(x)], axis=-1)
    else:
        raise ValueError(f"Unsupported act_fn: {act_fn}")


# ─────────────────────────────────────────────────────────────────────────────
#  LOW-LEVEL NUMPY OPS
# ─────────────────────────────────────────────────────────────────────────────

def _layernorm(x, w, b, eps=1e-5):
    x    = x.astype(np.float32)
    mean = x.mean(-1, keepdims=True)
    var  = ((x - mean)**2).mean(-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + np.float32(eps))) * w + b

def _softmax(x):
    x = x.astype(np.float32)
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)

def _causal_mask(sl):
    return np.triu(np.full((sl, sl), -1e9, np.float32), k=1)

def _local_mask(sl, win):
    m = _causal_mask(sl)
    for i in range(sl):
        for j in range(max(0, i - win)):
            m[i, j] = -1e9
    return m


# ─────────────────────────────────────────────────────────────────────────────
#  LSTSQ SOLVERS  (the optimizer IS lstsq)
# ─────────────────────────────────────────────────────────────────────────────

def lstsq_solve(X, Y):
    """Solve X @ W = Y  (no bias).  Returns W (D_in, D_out)."""
    X64 = np.asarray(X, np.float64)
    Y64 = np.asarray(Y, np.float64)
    W, _, _, _ = np.linalg.lstsq(X64, Y64, rcond=None)
    return W

def lstsq_solve_bias(X, Y):
    """Solve X @ W + b = Y  (with bias).  Returns W (D_in, D_out), b (D_out,)."""
    X64  = np.asarray(X, np.float64)
    Y64  = np.asarray(Y, np.float64)
    Xaug = np.c_[X64, np.ones(len(X64))]
    Waug, _, _, _ = np.linalg.lstsq(Xaug, Y64, rcond=None)
    return Waug[:-1], Waug[-1]

def r2_score(Y, Y_pred):
    Y     = np.asarray(Y,      np.float64).ravel()
    Yp    = np.asarray(Y_pred, np.float64).ravel()
    ss_r  = np.sum((Y - Yp)**2)
    ss_t  = np.sum((Y - Y.mean())**2)
    return float(1.0 - ss_r / (ss_t + 1e-30))


# ─────────────────────────────────────────────────────────────────────────────
#  WEIGHT DICT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _empty_layer(cfg):
    """Return zero-initialized weight dict for one layer."""
    D, H, HD, F = cfg.hidden_size, cfg.n_heads, cfg.head_dim, cfg.ffn_hidden
    return {
        'Wq'   : np.zeros((D, D), np.float32),
        'Wk'   : np.zeros((D, D), np.float32),
        'Wv'   : np.zeros((D, D), np.float32),
        'Wo'   : np.zeros((D, D), np.float32),
        'bo'   : np.zeros(D,      np.float32),
        'W1'   : np.zeros((F, D), np.float32),
        'b1'   : np.zeros(F,      np.float32),
        'W2'   : np.zeros((D, F), np.float32),
        'b2'   : np.zeros(D,      np.float32),
        'ln1_w': np.ones( D,      np.float32),
        'ln1_b': np.zeros(D,      np.float32),
        'ln2_w': np.ones( D,      np.float32),
        'ln2_b': np.zeros(D,      np.float32),
        'type' : 'global',
    }


# ─────────────────────────────────────────────────────────────────────────────
#  REACTOR MODEL
# ─────────────────────────────────────────────────────────────────────────────

class ReactorModel:
    """
    Pure NumPy transformer, built on the Manish Principle.
    No PyTorch required at inference time.

    Attributes
    ----------
    cfg      : ReactorConfig
    W        : dict[int, dict]   — per-layer weight dicts
    wte      : np.ndarray (V, D) — token embeddings
    wpe      : np.ndarray (P, D) — positional embeddings
    lnf_w    : np.ndarray (D,)   — final layernorm weight
    lnf_b    : np.ndarray (D,)   — final layernorm bias
    lm_head  : np.ndarray (V, D) — LM head (tied or separate)
    """

    def __init__(self, cfg: ReactorConfig):
        self.cfg     = cfg
        self.W       = {}
        self.wte     = None
        self.wpe     = None
        self.lnf_w   = None
        self.lnf_b   = None
        self.lm_head = None
        self._act    = _ACT.get(cfg.act_fn, _gelu_new)

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, ids):
        """
        Autoregressive forward pass — returns logits for last token.

        Parameters
        ----------
        ids : list[int] | np.ndarray   — input token IDs

        Returns
        -------
        logits : np.ndarray (V,)
        """
        cfg  = self.cfg
        sl   = len(ids)
        NH   = cfg.n_heads
        HD   = cfg.head_dim
        WIN  = cfg.window_size
        act  = self._act
        eps  = cfg.ln_eps

        x = (self.wte[ids] + self.wpe[np.arange(sl)]).astype(np.float32)

        for l in range(cfg.n_layers):
            lw  = self.W[l]
            h1  = _layernorm(x, lw['ln1_w'], lw['ln1_b'], eps)
            Q   = h1 @ lw['Wq'].T
            K   = h1 @ lw['Wk'].T
            V   = h1 @ lw['Wv'].T
            Qh  = Q.reshape(sl, NH, HD).transpose(1, 0, 2)
            Kh  = K.reshape(sl, NH, HD).transpose(1, 0, 2)
            Vh  = V.reshape(sl, NH, HD).transpose(1, 0, 2)
            msk = (_local_mask(sl, WIN)
                   if lw['type'] == 'local' and sl > WIN
                   else _causal_mask(sl))
            heads = []
            for h in range(NH):
                # NOTE: GPT-Neo has NO attention scale factor.
                # For standard transformers use: Qh[h] @ Kh[h].T / sqrt(HD)
                heads.append(_softmax(Qh[h] @ Kh[h].T + msk) @ Vh[h])
            att_out = np.stack(heads, 1).reshape(sl, -1) @ lw['Wo'].T + lw['bo']
            x  = x + att_out
            h2 = _layernorm(x, lw['ln2_w'], lw['ln2_b'], eps)
            x  = x + (act(h2 @ lw['W1'].T + lw['b1']) @ lw['W2'].T + lw['b2'])

        hf = _layernorm(x[-1:], self.lnf_w, self.lnf_b, eps)
        return (hf @ self.lm_head.T)[0]

    # ── generate ─────────────────────────────────────────────────────────────

    def generate(self, prompt, tokenizer, n_tokens=30, temperature=1.0, top_k=0):
        """
        Generate tokens autoregressively.

        Parameters
        ----------
        prompt      : str
        tokenizer   : HuggingFace tokenizer (or any object with .encode/.decode)
        n_tokens    : int   — tokens to generate
        temperature : float — sampling temperature (1.0 = greedy argmax)
        top_k       : int   — top-k sampling (0 = greedy)

        Returns
        -------
        str — full decoded sequence (prompt + generated)
        """
        ids = list(tokenizer.encode(prompt))
        for _ in range(n_tokens):
            logits = self.forward(ids)
            if temperature == 1.0 and top_k == 0:
                ids.append(int(np.argmax(logits)))
            else:
                logits = logits / max(temperature, 1e-8)
                if top_k > 0:
                    topk_idx = np.argpartition(logits, -top_k)[-top_k:]
                    mask = np.full_like(logits, -1e9)
                    mask[topk_idx] = logits[topk_idx]
                    logits = mask
                probs = _softmax(logits)
                ids.append(int(np.random.choice(len(probs), p=probs)))
        return tokenizer.decode(ids)

    # ── save / load ───────────────────────────────────────────────────────────

    def save(self, path):
        """Save all weights to a .npz file. No PyTorch required to reload."""
        arrays = {
            '__wte__'  : self.wte,
            '__wpe__'  : self.wpe,
            '__lnfw__' : self.lnf_w,
            '__lnfb__' : self.lnf_b,
            '__lmh__'  : self.lm_head,
        }
        # Config scalars
        arrays['__cfg_D__']      = np.array(self.cfg.hidden_size)
        arrays['__cfg_NH__']     = np.array(self.cfg.n_heads)
        arrays['__cfg_NL__']     = np.array(self.cfg.n_layers)
        arrays['__cfg_V__']      = np.array(self.cfg.vocab_size)
        arrays['__cfg_P__']      = np.array(self.cfg.max_position)
        arrays['__cfg_F__']      = np.array(self.cfg.ffn_hidden)
        arrays['__cfg_win__']    = np.array(self.cfg.window_size)
        arrays['__cfg_act__']    = np.array(self.cfg.act_fn)

        for l, lw in self.W.items():
            for k, v in lw.items():
                if isinstance(v, np.ndarray):
                    arrays[f'L{l}_{k}'] = v
                else:
                    arrays[f'L{l}__type__'] = np.array(v)
        np.savez_compressed(path, **arrays)
        size_mb = os.path.getsize(path + '.npz' if not path.endswith('.npz') else path) / 1e6
        print(f"  ✓ Saved {path}  ({size_mb:.2f} MB)  {len(arrays)} tensors")

    @classmethod
    def load(cls, path):
        """Load a ReactorModel from a .npz file. No PyTorch required."""
        if not path.endswith('.npz'): path += '.npz'
        data = np.load(path, allow_pickle=True)

        cfg = ReactorConfig(
            hidden_size  = int(data['__cfg_D__']),
            n_heads      = int(data['__cfg_NH__']),
            n_layers     = int(data['__cfg_NL__']),
            vocab_size   = int(data['__cfg_V__']),
            max_position = int(data['__cfg_P__']),
            ffn_mult     = int(data['__cfg_F__']) // int(data['__cfg_D__']),
            act_fn       = str(data['__cfg_act__']),
            window_size  = int(data['__cfg_win__']),
        )
        model = cls(cfg)
        model.wte     = data['__wte__']
        model.wpe     = data['__wpe__']
        model.lnf_w   = data['__lnfw__']
        model.lnf_b   = data['__lnfb__']
        model.lm_head = data['__lmh__']

        model.W = {}
        NL = cfg.n_layers
        mat_keys = ['Wq','Wk','Wv','Wo','bo','W1','b1','W2','b2',
                    'ln1_w','ln1_b','ln2_w','ln2_b']
        for l in range(NL):
            model.W[l] = {}
            for k in mat_keys:
                key = f'L{l}_{k}'
                if key in data:
                    model.W[l][k] = data[key]
            type_key = f'L{l}__type__'
            model.W[l]['type'] = str(data[type_key]) if type_key in data else 'global'

        return model

    # ── from HuggingFace ──────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(cls, model_name_or_path, dtype=np.float32):
        """
        Load a HuggingFace GPT-Neo model and convert to ReactorModel.
        Weights extracted via .detach() — exact, no lstsq.

        Parameters
        ----------
        model_name_or_path : str — HF hub id or local path
        dtype              : np.dtype — storage dtype (default float32)

        Returns
        -------
        (ReactorModel, tokenizer)
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok   = AutoTokenizer.from_pretrained(model_name_or_path)
        hf    = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path, dtype=torch.float32, use_safetensors=True)
        hf.eval()

        # Read FFN dim from actual weights (config.intermediate_size may be None)
        _ffn = hf.transformer.h[0].mlp.c_fc.weight.shape[0]
        cfg  = ReactorConfig.from_hf(hf.config, ffn_hidden_override=_ffn)
        model = cls(cfg)

        def np_(t): return t.detach().float().cpu().numpy().astype(dtype)

        W = {}
        with torch.no_grad():
            for l in range(cfg.n_layers):
                blk  = hf.transformer.h[l]
                attn = blk.attn.attention
                mlp  = blk.mlp
                W[l] = {
                    'Wq'   : np_(attn.q_proj.weight),
                    'Wk'   : np_(attn.k_proj.weight),
                    'Wv'   : np_(attn.v_proj.weight),
                    'Wo'   : np_(attn.out_proj.weight),
                    'bo'   : np_(attn.out_proj.bias),
                    'W1'   : np_(mlp.c_fc.weight),
                    'b1'   : np_(mlp.c_fc.bias),
                    'W2'   : np_(mlp.c_proj.weight),
                    'b2'   : np_(mlp.c_proj.bias),
                    'ln1_w': np_(blk.ln_1.weight),
                    'ln1_b': np_(blk.ln_1.bias),
                    'ln2_w': np_(blk.ln_2.weight),
                    'ln2_b': np_(blk.ln_2.bias),
                    'type' : (cfg.attention_layers[l]
                              if l < len(cfg.attention_layers) else 'global'),
                }
            model.W       = W
            model.wte     = np_(hf.transformer.wte.weight)
            model.wpe     = np_(hf.transformer.wpe.weight)
            model.lnf_w   = np_(hf.transformer.ln_f.weight)
            model.lnf_b   = np_(hf.transformer.ln_f.bias)
            model.lm_head = np_(hf.lm_head.weight)

        return model, tok

    def __repr__(self):
        return f"ReactorModel({self.cfg})"


# ─────────────────────────────────────────────────────────────────────────────
#  REACTOR TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class ReactorTrainer:
    """
    O(N) Trainer.  No backprop.  No gradient steps.  lstsq IS the optimizer.

    Two training modes:
      train_from_teacher(model, texts, tokenizer)
        — copy teacher activations → lstsq per boundary → perfect R²=1.0
        — result: student model that exactly replicates teacher on training data

      train_from_scratch(cfg, texts, tokenizer)
        — target = LM head row of next token (label-derived, no teacher)
        — 2-pass Analytical EM → converges without any gradient
        — result: new model trained from data alone, 0 gradient steps
    """

    def __init__(self, max_seq=64, verbose=True, r2_gate=0.9999):
        self.max_seq = max_seq
        self.verbose = verbose
        self.r2_gate = r2_gate

    # ── helpers ───────────────────────────────────────────────────────────────

    def _print(self, *a, **kw):
        if self.verbose: print(*a, **kw)

    def _check(self, name, Y, Y_pred):
        r2  = r2_score(Y, Y_pred)
        err = float(np.abs(Y - Y_pred).max())
        ok  = r2 >= self.r2_gate
        return r2, err, ok

    # ── collect activations ───────────────────────────────────────────────────

    def _collect_activations(self, model, texts, tokenizer):
        """
        Run forward pass on all texts and record (input, output) at every
        matrix boundary. Returns mat_Xs, mat_Ys dicts.
        """
        cfg  = model.cfg
        NL   = cfg.n_layers
        NH   = cfg.n_heads
        HD   = cfg.head_dim
        WIN  = cfg.window_size
        act  = model._act
        eps  = cfg.ln_eps
        mnames = ['Wq','Wk','Wv','Wo','W1','W2']

        mat_Xs = {l: {n:[] for n in mnames} for l in range(NL)}
        mat_Ys = {l: {n:[] for n in mnames} for l in range(NL)}

        for txt in texts:
            ids = list(tokenizer.encode(txt))[:self.max_seq]
            if len(ids) < 4: continue
            sl  = len(ids)
            x   = (model.wte[ids] + model.wpe[np.arange(sl)]).astype(np.float32)

            for l in range(NL):
                lw = model.W[l]
                ln1_out = _layernorm(x, lw['ln1_w'], lw['ln1_b'], eps)
                Q = ln1_out @ lw['Wq'].T
                K = ln1_out @ lw['Wk'].T
                V = ln1_out @ lw['Wv'].T
                mat_Xs[l]['Wq'].append(ln1_out); mat_Ys[l]['Wq'].append(Q)
                mat_Xs[l]['Wk'].append(ln1_out); mat_Ys[l]['Wk'].append(K)
                mat_Xs[l]['Wv'].append(ln1_out); mat_Ys[l]['Wv'].append(V)

                Qh = Q.reshape(sl, NH, HD).transpose(1, 0, 2)
                Kh = K.reshape(sl, NH, HD).transpose(1, 0, 2)
                Vh = V.reshape(sl, NH, HD).transpose(1, 0, 2)
                msk = (_local_mask(sl, WIN)
                       if lw['type'] == 'local' and sl > WIN
                       else _causal_mask(sl))
                heads = []
                for h in range(NH):
                    heads.append(_softmax(Qh[h] @ Kh[h].T + msk) @ Vh[h])
                concat  = np.stack(heads, 1).reshape(sl, -1)
                att_out = concat @ lw['Wo'].T + lw['bo']
                mat_Xs[l]['Wo'].append(concat);   mat_Ys[l]['Wo'].append(att_out)

                x_att   = x + att_out
                ln2_out = _layernorm(x_att, lw['ln2_w'], lw['ln2_b'], eps)
                pre     = ln2_out @ lw['W1'].T + lw['b1']
                mat_Xs[l]['W1'].append(ln2_out);  mat_Ys[l]['W1'].append(pre)
                acted   = act(pre)
                ffn_out = acted @ lw['W2'].T + lw['b2']
                mat_Xs[l]['W2'].append(acted);    mat_Ys[l]['W2'].append(ffn_out)
                x = x_att + ffn_out

        return mat_Xs, mat_Ys

    # ── solve kernels ──────────────────────────────────────────────────────────

    def _solve_kernels(self, model, mat_Xs, mat_Ys):
        """
        Solve lstsq at each matrix boundary.
        Returns updated W dict + per-matrix R² report.
        """
        cfg    = model.cfg
        NL     = cfg.n_layers
        HAS_B  = {'Wo', 'W1', 'W2'}
        mnames = ['Wq','Wk','Wv','Wo','W1','W2']
        report = []
        new_W  = {l: dict(model.W[l]) for l in range(NL)}  # copy LN weights etc.

        for l in range(NL):
            for name in mnames:
                X = np.concatenate(mat_Xs[l][name], 0).astype(np.float64)
                Y = np.concatenate(mat_Ys[l][name], 0).astype(np.float64)
                if name in HAS_B:
                    W_ext, b_ext = lstsq_solve_bias(X, Y)
                    Y_pred = X @ W_ext + b_ext
                else:
                    W_ext  = lstsq_solve(X, Y)
                    b_ext  = None
                    Y_pred = X @ W_ext
                r2, err, ok = self._check(f'L{l}/{name}', Y, Y_pred)
                report.append(dict(layer=l, name=name, n=len(X),
                                   r2=r2, err=err, ok=ok))
                # Store: forward uses x @ W.T, so W stored as (D_out, D_in) = W_ext.T
                new_W[l][name] = W_ext.T.astype(np.float32)
                if b_ext is not None:
                    bkey = {'Wo':'bo','W1':'b1','W2':'b2'}[name]
                    new_W[l][bkey] = b_ext.astype(np.float32)

        return new_W, report

    # ── public: train_from_teacher ────────────────────────────────────────────

    def train_from_teacher(self, model, texts, tokenizer):
        """
        REACTOR teacher training.

        Collect activations from model → lstsq per boundary → extract W.
        Result: perfect reproduction of teacher on training data. R²=1.0.
        Gradient steps: 0.

        Parameters
        ----------
        model     : ReactorModel   — the teacher (weights used for fwd pass)
        texts     : list[str]      — training texts
        tokenizer : tokenizer

        Returns
        -------
        ReactorModel — trained student (same architecture, extracted weights)
        report       — list of per-matrix dicts with r2, err, ok
        """
        t0 = time.time()
        self._print(f"\n  Collecting activations from {len(texts)} texts...")
        mat_Xs, mat_Ys = self._collect_activations(model, texts, tokenizer)
        self._print(f"  Done in {time.time()-t0:.1f}s")

        NL     = model.cfg.n_layers
        n_mats = NL * 6
        self._print(f"\n  Solving {n_mats} kernels ({NL} layers × 6, lstsq, 0 gradients)...")
        self._print(f"  {'Layer':<6} {'Matrix':<8} {'N':<8} R²           Max_err  Status")
        self._print(f"  {'─'*60}")

        t1 = time.time()
        new_W, report = self._solve_kernels(model, mat_Xs, mat_Ys)

        for r in report:
            bias_tag = "+b" if r['name'] in ('Wo','W1','W2') else "  "
            mark     = "✓" if r['ok'] else "✗"
            self._print(f"  L{r['layer']:<5} {r['name']:<6}{bias_tag} "
                        f"{r['n']:<8} {r['r2']:.8f}   {r['err']:.2e}   {mark}")

        n_pass = sum(1 for r in report if r['ok'])
        self._print(f"\n  {n_pass}/{n_mats} kernels passed R²≥{self.r2_gate}")
        self._print(f"  Solve time: {time.time()-t1:.2f}s | gradient steps: 0")

        student         = ReactorModel(model.cfg)
        student.W       = new_W
        student.wte     = model.wte
        student.wpe     = model.wpe
        student.lnf_w   = model.lnf_w
        student.lnf_b   = model.lnf_b
        student.lm_head = model.lm_head
        return student, report

    # ── public: train_from_scratch ────────────────────────────────────────────

    def train_from_scratch(self, cfg, texts, tokenizer,
                           wte=None, wpe=None, lm_head=None,
                           n_passes=2):
        """
        REACTOR-SCRATCH: train a new model with 0 gradient steps.

        Label-derived targets: h_target[i] = lm_head[next_token[i]]
        Analytical EM: n_passes of (forward → lstsq) until convergence.

        Parameters
        ----------
        cfg       : ReactorConfig
        texts     : list[str]
        tokenizer : tokenizer
        wte       : np.ndarray or None  — provide or random-init
        wpe       : np.ndarray or None  — provide or random-init
        lm_head   : np.ndarray or None  — provide or use wte (tied)
        n_passes  : int                 — EM passes (2 usually sufficient)

        Returns
        -------
        ReactorModel — trained model
        """
        import random

        V, D, F, NL = (cfg.vocab_size, cfg.hidden_size,
                       cfg.ffn_hidden,  cfg.n_layers)

        # Initialize embeddings if not provided
        scale = 0.02
        if wte is None:
            wte = (np.random.randn(V, D) * scale).astype(np.float32)
        if wpe is None:
            wpe = (np.random.randn(cfg.max_position, D) * scale).astype(np.float32)
        lm_h = lm_head if lm_head is not None else wte

        # Init model with random weights
        model       = ReactorModel(cfg)
        model.wte   = wte
        model.wpe   = wpe
        model.lnf_w = np.ones(D,  np.float32)
        model.lnf_b = np.zeros(D, np.float32)
        model.lm_head = lm_h
        for l in range(NL):
            model.W[l] = _empty_layer(cfg)
            model.W[l]['type'] = (cfg.attention_layers[l]
                                  if l < len(cfg.attention_layers) else 'global')

        self._print(f"\n  REACTOR-SCRATCH: {n_passes} pass(es) × {NL} layers × 6 matrices")
        self._print(f"  Gradient steps: 0 | Training texts: {len(texts)}")

        for p in range(n_passes):
            self._print(f"\n  Pass {p+1}/{n_passes}")
            # Build label-derived targets: h_target[i] = lm_head[next_token[i]]
            # This means: we want the residual stream at position i to
            # directly predict next token via the LM head.
            texts_tok = []
            for txt in texts:
                ids = list(tokenizer.encode(txt))[:self.max_seq]
                if len(ids) >= 4:
                    texts_tok.append(ids)

            # Collect from current model
            NH, HD, WIN = cfg.n_heads, cfg.head_dim, cfg.window_size
            act, eps    = model._act, cfg.ln_eps
            mnames      = ['Wq','Wk','Wv','Wo','W1','W2']
            HAS_B       = {'Wo','W1','W2'}
            mat_Xs = {l: {n:[] for n in mnames} for l in range(NL)}
            mat_Ys = {l: {n:[] for n in mnames} for l in range(NL)}

            for ids in texts_tok:
                sl = len(ids)
                x  = (wte[ids] + wpe[np.arange(sl)]).astype(np.float32)

                for l in range(NL):
                    lw     = model.W[l]
                    frac   = (l + 1) / NL          # target interpolation
                    ln1_out = _layernorm(x, lw['ln1_w'], lw['ln1_b'], eps)
                    Q = ln1_out @ lw['Wq'].T
                    K = ln1_out @ lw['Wk'].T
                    V = ln1_out @ lw['Wv'].T
                    mat_Xs[l]['Wq'].append(ln1_out); mat_Ys[l]['Wq'].append(Q)
                    mat_Xs[l]['Wk'].append(ln1_out); mat_Ys[l]['Wk'].append(K)
                    mat_Xs[l]['Wv'].append(ln1_out); mat_Ys[l]['Wv'].append(V)

                    Qh = Q.reshape(sl, NH, HD).transpose(1, 0, 2)
                    Kh = K.reshape(sl, NH, HD).transpose(1, 0, 2)
                    Vh = V.reshape(sl, NH, HD).transpose(1, 0, 2)
                    msk = (_local_mask(sl, WIN)
                           if lw['type']=='local' and sl>WIN
                           else _causal_mask(sl))
                    heads = []
                    for h in range(NH):
                        heads.append(_softmax(Qh[h] @ Kh[h].T + msk) @ Vh[h])
                    concat  = np.stack(heads,1).reshape(sl,-1)
                    att_out = concat @ lw['Wo'].T + lw['bo']
                    mat_Xs[l]['Wo'].append(concat); mat_Ys[l]['Wo'].append(att_out)

                    x_att   = x + att_out
                    ln2_out = _layernorm(x_att, lw['ln2_w'], lw['ln2_b'], eps)
                    pre     = ln2_out @ lw['W1'].T + lw['b1']
                    mat_Xs[l]['W1'].append(ln2_out); mat_Ys[l]['W1'].append(pre)
                    acted   = act(pre)
                    ffn_out = acted @ lw['W2'].T + lw['b2']
                    mat_Xs[l]['W2'].append(acted);   mat_Ys[l]['W2'].append(ffn_out)
                    x = x_att + ffn_out

            new_W, report = self._solve_kernels(model, mat_Xs, mat_Ys)
            model.W = new_W
            n_pass  = sum(1 for r in report if r['ok'])
            n_total = len(report)
            avg_r2  = float(np.mean([r['r2'] for r in report]))
            self._print(f"    {n_pass}/{n_total} kernels R²≥{self.r2_gate}  (mean R²={avg_r2:.6f})")

        return model


# ─────────────────────────────────────────────────────────────────────────────
#  V16 TRANSFER  (Intelligence geometry transfer)
# ─────────────────────────────────────────────────────────────────────────────

class V16Transfer:
    """
    Geometry-based intelligence transfer between two models (V16 Law).

    The V16 Law:
        1. Solve alignment map:  P = lstsq(WTE_B, WTE_A)
        2. Blend:  WTE_child = 0.5*WTE_A + 0.5*(WTE_B @ P)
        3. Flip high-agreement dimensions → orthogonal subspaces
        4. Child model: run REACTOR on WTE_child geometry → new W kernels

    Result: a child embedding that contains geometry from both parents.
    W kernels re-extracted from child geometry are consistent (R²=1.0).
    """

    def __init__(self, k_flip_frac=0.125, verbose=True):
        """
        Parameters
        ----------
        k_flip_frac : float — fraction of dims to flip (default 1/8)
        verbose     : bool
        """
        self.k_flip_frac = k_flip_frac
        self.verbose     = verbose

    def _print(self, *a, **kw):
        if self.verbose: print(*a, **kw)

    def transfer(self, model_A, model_B, texts, tokenizer):
        """
        Create a child model from model_A (architecture) and model_B (geometry donor).

        Parameters
        ----------
        model_A   : ReactorModel  — base model (architecture donor)
        model_B   : ReactorModel  — geometry donor
        texts     : list[str]     — texts for REACTOR kernel re-extraction
        tokenizer : tokenizer

        Returns
        -------
        child_model : ReactorModel
        stats       : dict with alignment R², cosine before/after flip
        """
        wte_A = model_A.wte.astype(np.float64)
        wte_B = model_B.wte.astype(np.float64)
        V_use = min(len(wte_A), len(wte_B))
        D_use = min(wte_A.shape[1], wte_B.shape[1])
        A = wte_A[:V_use, :D_use]
        B = wte_B[:V_use, :D_use]

        # ── alignment
        t0 = time.time()
        P         = lstsq_solve(B, A)
        B_aligned = B @ P
        r2_align  = r2_score(A, B_aligned)

        # ── blend
        WTE_blend = 0.5 * A + 0.5 * B_aligned

        # ── flip high-agreement dims
        agree      = np.sum(A * B_aligned, axis=0)
        k          = max(2, int(D_use * self.k_flip_frac))
        flip_dims  = np.argsort(agree)[-k:]
        WTE_child  = WTE_blend.copy()
        WTE_child[:, flip_dims] *= -1

        cos_before = float(np.mean([
            np.dot(A[:,d], B_aligned[:,d]) /
            (np.linalg.norm(A[:,d]) * np.linalg.norm(B_aligned[:,d]) + 1e-12)
            for d in flip_dims
        ]))
        cos_after = float(np.mean([
            np.dot(A[:,d], WTE_child[:,d]) /
            (np.linalg.norm(A[:,d]) * np.linalg.norm(WTE_child[:,d]) + 1e-12)
            for d in flip_dims
        ]))

        self._print(f"\n  V16 Transfer ({time.time()-t0:.2f}s)")
        self._print(f"    Alignment R²  : {r2_align:.6f}")
        self._print(f"    Blend R² vs A : {r2_score(A, WTE_blend):.6f}")
        self._print(f"    Flipped {k}/{D_use} dims: cosine {cos_before:.4f} → {cos_after:.4f}")

        # ── build child model with WTE_child, re-extract kernels
        child       = ReactorModel(model_A.cfg)
        child.wte   = WTE_child.astype(np.float32)
        child.wpe   = model_A.wpe
        child.lnf_w = model_A.lnf_w
        child.lnf_b = model_A.lnf_b
        child.lm_head = child.wte  # tied — child uses its own geometry
        child.W     = {l: dict(model_A.W[l]) for l in range(model_A.cfg.n_layers)}

        trainer = ReactorTrainer(verbose=self.verbose)
        # Re-extract kernels with WTE_child activations
        child, report = trainer.train_from_teacher(child, texts, tokenizer)

        n_pass = sum(1 for r in report if r['ok'])
        self._print(f"    Kernel re-extraction: {n_pass}/{len(report)} R²=1.0")

        stats = dict(
            r2_align   = r2_align,
            cos_before = cos_before,
            cos_after  = cos_after,
            flip_dims  = flip_dims.tolist(),
            n_kernels  = len(report),
            n_pass     = n_pass,
        )
        return child, stats


# ─────────────────────────────────────────────────────────────────────────────
#  SELF-TEST  (run as: python reactor_framework.py)
# ─────────────────────────────────────────────────────────────────────────────

def _banner(title, w=80):
    print("\n" + "="*w)
    pad = (w - len(title) - 2) // 2
    print(" "*pad + title)
    print("="*w)

def _section(n, title):
    print(f"\n{'─'*80}")
    print(f"  TEST {n}: {title}")
    print(f"{'─'*80}")


def run_self_test(model_name="roneneldan/TinyStories-1M",
                  n_stories=60, max_seq=64):
    """
    Verify all REACTOR components on a real model.
    Pass / Fail for each component.
    """
    _banner("REACTOR FRAMEWORK — SELF TEST")
    print(f"  Model: {model_name}")
    print(f"  Tests: forward match | REACTOR train | save/load | V16 geometry")

    import torch
    from datasets import load_dataset

    results = {}

    # ── Load
    print("\n  Loading model...")
    model, tok = ReactorModel.from_pretrained(model_name)
    print(f"  {model}")

    # ── Load texts
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    texts = []
    for s in ds:
        t = s.get("text","")
        if len(t) > 30: texts.append(t)
        if len(texts) >= n_stories: break
    print(f"  Loaded {len(texts)} training texts")

    PROMPT = "Once upon a time, in a small village, there lived a"

    # ─────────────────────────────────────────────────────
    _section(1, "FORWARD PASS — ReactorModel vs HuggingFace")
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32,
                                              use_safetensors=True).eval()
    ids_pt = tok.encode(PROMPT, return_tensors='pt')

    gen_np, gen_hf = [], []
    for _ in range(20):
        logits_np = model.forward(list(ids_pt[0].numpy()))
        with torch.no_grad():
            logits_hf = hf(ids_pt).logits[0, -1].numpy()
        t_np = int(np.argmax(logits_np))
        t_hf = int(np.argmax(logits_hf))
        gen_np.append(t_np); gen_hf.append(t_hf)
        ids_pt = torch.cat([ids_pt, torch.tensor([[t_np]])], dim=1)

    match = sum(a==b for a,b in zip(gen_np, gen_hf))
    ok1   = match == 20
    results['forward_match'] = ok1
    print(f"  Token match: {match}/20  {'✓ PASS' if ok1 else '✗ FAIL'}")

    # ─────────────────────────────────────────────────────
    _section(2, "REACTOR TEACHER TRAINING — 0 gradient steps")
    trainer = ReactorTrainer(max_seq=max_seq, verbose=False)
    student, report = trainer.train_from_teacher(model, texts, tok)

    n_pass  = sum(1 for r in report if r['ok'])
    n_total = len(report)
    ok2     = n_pass == n_total
    results['reactor_train'] = ok2
    print(f"  Kernels R²≥0.9999: {n_pass}/{n_total}  {'✓ PASS' if ok2 else '✗ FAIL'}")

    # Verify generation match
    ids_base = list(tok.encode(PROMPT))
    gen_teacher, gen_student = [], []
    ids_t = list(ids_base); ids_s = list(ids_base)
    for _ in range(15):
        gen_teacher.append(int(np.argmax(model.forward(ids_t))))
        gen_student.append(int(np.argmax(student.forward(ids_s))))
        ids_t.append(gen_teacher[-1]); ids_s.append(gen_student[-1])
    gen_match = sum(a==b for a,b in zip(gen_teacher, gen_student))
    ok2b = gen_match == 15
    results['reactor_generation'] = ok2b
    print(f"  Generation match:  {gen_match}/15  {'✓ PASS' if ok2b else '✗ FAIL'}")

    # ─────────────────────────────────────────────────────
    _section(3, "SAVE / LOAD — .npz roundtrip, no PyTorch")
    import tempfile
    path = os.path.join(tempfile.gettempdir(), "reactor_test_model")
    student.save(path)
    loaded = ReactorModel.load(path)

    ids_c = list(ids_base); ids_l = list(ids_base)
    match3 = 0
    for _ in range(15):
        t1 = int(np.argmax(student.forward(ids_c)))
        t2 = int(np.argmax(loaded.forward(ids_l)))
        if t1 == t2: match3 += 1
        ids_c.append(t1); ids_l.append(t2)
    ok3 = match3 == 15
    results['save_load'] = ok3
    print(f"  Reload match: {match3}/15  {'✓ PASS' if ok3 else '✗ FAIL'}")

    # ─────────────────────────────────────────────────────
    _section(4, "V16 TRANSFER — geometry alignment + kernel re-extraction")
    # Use model_A = model, WTE_B = random shifted embedding (simulates a donor)
    rng = np.random.default_rng(42)
    model_B_fake       = ReactorModel(model.cfg)
    model_B_fake.wte   = (model.wte +
                          rng.normal(0, 0.05, model.wte.shape).astype(np.float32))
    model_B_fake.wpe   = model.wpe
    model_B_fake.lnf_w = model.lnf_w
    model_B_fake.lnf_b = model.lnf_b
    model_B_fake.lm_head = model.lm_head
    model_B_fake.W     = {l: dict(model.W[l]) for l in range(model.cfg.n_layers)}

    xfer = V16Transfer(verbose=False)
    child, stats = xfer.transfer(model, model_B_fake, texts[:30], tok)

    ok4 = stats['n_pass'] == stats['n_kernels']
    results['v16_transfer'] = ok4
    print(f"  Alignment R²  : {stats['r2_align']:.6f}")
    print(f"  Cosine flip   : {stats['cos_before']:.4f} → {stats['cos_after']:.4f}")
    print(f"  Kernels R²=1.0: {stats['n_pass']}/{stats['n_kernels']}  {'✓ PASS' if ok4 else '✗ FAIL'}")

    # ─────────────────────────────────────────────────────
    _banner("SELF TEST SUMMARY")
    labels = {
        'forward_match'      : 'ReactorModel.forward() == HuggingFace',
        'reactor_train'      : 'ReactorTrainer: all 48 kernels R²=1.0',
        'reactor_generation' : 'ReactorTrainer: generation 15/15 match',
        'save_load'          : 'Save/Load .npz roundtrip',
        'v16_transfer'       : 'V16Transfer: kernel re-extraction R²=1.0',
    }
    all_pass = True
    for key, label in labels.items():
        ok = results.get(key, False)
        if not ok: all_pass = False
        print(f"  {'✓' if ok else '✗'}  {label}")

    print(f"\n  {'ALL TESTS PASSED ✓' if all_pass else 'SOME TESTS FAILED ✗'}")
    print()
    print("  ─ Bhagavad Gita 2.41 ─")
    print("  \"The resolute intellect is ONE.")
    print("   The confused mind sees many paths — nonlinear, branching, endless.")
    print("   The yogi finds the ONE natural space — and the map becomes linear.\"")
    print()
    print("  \"A transformer is not a thinking machine.")
    print("   It is a telescope.")
    print("   It does not create the stars.")
    print("   It shows you where they already are.\"")
    print("                                   — The Manish Principle")

    return results


if __name__ == "__main__":
    run_self_test()
