"""
Anima Artist Blend — hidden-state blending for Qwen3 0.6B text encoder.

Problem
-------
Putting several artist tags into one prompt string does not blend them:
the tags compete inside a single encoding pass and one artist wins.
This is worse with a small causal-LM encoder such as Qwen3 0.6B
(the Anima text encoder): later tokens can see earlier ones but not the
reverse, so the "mix" degenerates into an order fight, and training
captions almost never contained more than one artist tag anyway.

Approach (v0.1)
---------------
Never put two artist tags in the same encoding pass.

* ``pooled_token`` mode (default, the experiment):
  Encode ``base_prompt`` alone and ``base_prompt + artist_i`` for each
  artist separately. Because Qwen is causal, the hidden states of the
  shared base prefix are (up to padding) identical across passes, so the
  base part is taken once. Each artist's trailing token span is mean
  pooled into a single vector, the vectors are blended with sharpened
  normalised weights, the norm is restored, and the result is appended
  to the base sequence as ONE virtual artist token. Downstream there is
  nothing left to compete.

  ``deviation_gain`` (v0.2): pooled artist vectors share a large common
  component (roughly "an artist tag is here"), while each artist's own
  signature lives in the small deviation from the cross-artist mean.
  Before blending, each vector's deviation from that mean is multiplied
  by this gain, so swapping or re-weighting artists actually moves the
  result. 1.0 reproduces the v0.1 behaviour. Pod tests (2026-11-05)
  showed nardack<->fuzichoco swaps were nearly invisible at gain 1.

* ``full_average`` mode (baseline for A/B testing):
  Encode each ``base_prompt + artist_i`` and take a norm-restored
  weighted average of the whole sequences (zero-padded to the longest).
  This is the classic conditioning-average behaviour, kept so the test
  can compare the two strategies with the same seed.

Assumptions to verify on a real install (private test):
* ``clip.encode_from_tokens_scheduled(clip.tokenize(text))`` returns
  ``[[tensor, dict], ...]`` as in stock CLIPTextEncode.
* The artist span is ``tensor[:, base_len:, :]`` where ``base_len`` is
  the token length of the base prompt encoded alone. Tokenisers that
  append EOS may include it in the span; its hidden state summarises
  the sequence, which is acceptable for a pooled style vector, but the
  printed span sizes should be checked on the first run.
"""

import torch

MAX_ARTISTS = 8


def sharpened_weights(raw_weights, sharpen):
    """Apply the sharpening exponent and normalise to sum 1.

    Pure python (no torch) so the weight maths can be unit tested
    outside ComfyUI. ``sharpen`` > 1 makes strong artists stronger and
    weak ones weaker, countering the "average grey style" collapse when
    many (5+) artists are blended at once.
    """
    powered = [max(float(w), 0.0) ** float(sharpen) for w in raw_weights]
    total = sum(powered)
    if total <= 0:
        return [1.0 / len(powered)] * len(powered) if powered else []
    return [w / total for w in powered]


def _encode(clip, text):
    """Encode one string and return (tensor, extra_dict) of entry 0."""
    tokens = clip.tokenize(text)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    tensor, extra = conditioning[0]
    return tensor, dict(extra) if extra else {}


def _renormalise(blended, target_norm):
    """Rescale the last-dim norm of ``blended`` to ``target_norm``."""
    norm = blended.norm(dim=-1, keepdim=True)
    scale = torch.where(norm > 0, target_norm / torch.clamp(norm, min=1e-12),
                          torch.ones_like(norm))
    return blended * scale


def _blend_vectors(vectors, weights):
    """Weighted blend of same-shaped vectors with norm restoration.

    vectors: list of tensors [1, 1, D] (or [1, D] for pooled outputs)
    weights: normalised weights (sum 1)
    Returns a tensor of the same shape whose last-dim norm is the
    weighted average of the input norms, so the blend does not fade.
    """
    blended = None
    target_norm = None
    for vec, w in zip(vectors, weights):
        term = vec * float(w)
        blended = term if blended is None else blended + term
        n = vec.norm(dim=-1, keepdim=True) * float(w)
        target_norm = n if target_norm is None else target_norm + n
    return _renormalise(blended, target_norm)


def _blend_pooled(extras, weights):
    """Blend 'pooled_output' across encoding results when present.

    An encoder with no pooled output (Anima's Qwen3 TE on ComfyUI 0.44)
    still carries the key with a None value, so None counts as absent.
    """
    vecs = [e["pooled_output"] for e in extras
            if e.get("pooled_output") is not None]
    if not vecs or len(vecs) != len(extras):
        return None
    # pooled outputs may be [1, D]; give _blend_vectors a 3D view.
    shaped = [v.unsqueeze(1) if v.dim() == 2 else v for v in vecs]
    blended = _blend_vectors(shaped, weights)
    return blended[:, 0] if vecs[0].dim() == 2 else blended


class AnimaArtistHiddenBlend:
    """Blend 1..8 artist tags in Qwen hidden-state space (Anima)."""

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "clip": ("CLIP",),
            "base_prompt": ("STRING", {"multiline": True, "default": ""}),
            "blend_mode": (["pooled_token", "full_average"],),
            "sharpen": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 3.0,
                                  "step": 0.1}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0,
                                   "step": 0.05}),
        }
        for i in range(1, MAX_ARTISTS + 1):
            required[f"artist_{i}"] = ("STRING", {"default": ""})
            required[f"weight_{i}"] = ("FLOAT", {"default": 1.0, "min": 0.0,
                                                 "max": 3.0, "step": 0.05})
        # Appended last on purpose: ComfyUI restores saved widget values
        # positionally, so a new input in the middle would shift every
        # value after it in workflows saved with older versions.
        required["deviation_gain"] = ("FLOAT", {"default": 2.0, "min": 1.0,
                                                "max": 5.0, "step": 0.25})
        return {"required": required}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "blend"
    CATEGORY = "conditioning/anima"

    def blend(self, clip, base_prompt, blend_mode, sharpen, strength,
              deviation_gain=2.0, **kwargs):
        artists, raw_weights = [], []
        for i in range(1, MAX_ARTISTS + 1):
            tag = (kwargs.get(f"artist_{i}") or "").strip()
            if tag:
                artists.append(tag)
                raw_weights.append(kwargs.get(f"weight_{i}", 1.0))

        if clip is None:
            raise RuntimeError("clip input is invalid: None")

        base_prompt = base_prompt or ""

        # No artist -> behave exactly like a plain CLIP text encode.
        if not artists:
            tensor, extra = _encode(clip, base_prompt)
            return ([[tensor, extra]],)

        weights = sharpened_weights(raw_weights, sharpen)

        def full_text(artist):
            if base_prompt.strip():
                return base_prompt.rstrip() + ", " + artist
            return artist

        if blend_mode == "full_average":
            return (self._full_average(clip, artists, weights, full_text,
                                       strength),)

        return (self._pooled_token(clip, base_prompt, artists, weights,
                                   full_text, strength, deviation_gain),)

    # -- mode 1: pooled virtual artist token -------------------------
    def _pooled_token(self, clip, base_prompt, artists, weights, full_text,
                      strength, deviation_gain=2.0):
        base_tensor, base_extra = _encode(clip, base_prompt)
        base_len = base_tensor.shape[1]

        pooled_vecs, extras = [], []
        for artist in artists:
            tensor, extra = _encode(clip, full_text(artist))
            extras.append(extra)
            if tensor.shape[1] > base_len:
                span = tensor[:, base_len:, :]
            else:
                # Fallback: tokeniser merged things unexpectedly; use the
                # last token rather than failing the whole graph.
                span = tensor[:, -1:, :]
                print(f"[AnimaArtistBlend] WARNING: '{artist}' produced no "
                      f"extra tokens (len={tensor.shape[1]}, base_len="
                      f"{base_len}); using last token as its span.")
            print(f"[AnimaArtistBlend] '{artist}': span tokens = "
                  f"{span.shape[1]} (total {tensor.shape[1]}, base "
                  f"{base_len})")
            pooled_vecs.append(span.mean(dim=1, keepdim=True))

        if deviation_gain != 1.0 and len(pooled_vecs) > 1:
            stacked = torch.stack(pooled_vecs, dim=0)
            mean_vec = stacked.mean(dim=0)
            dev_ratio = ((stacked - mean_vec).norm(dim=-1)
                         / mean_vec.norm(dim=-1).clamp(min=1e-12)).mean()
            print(f"[AnimaArtistBlend] deviation gain "
                  f"{deviation_gain}: artist deviation is "
                  f"{float(dev_ratio) * 100:.1f}% of the shared component")
            pooled_vecs = [mean_vec + (v - mean_vec) * float(deviation_gain)
                           for v in pooled_vecs]

        blended = _blend_vectors(pooled_vecs, weights) * float(strength)
        final = torch.cat([base_tensor, blended], dim=1)

        out_extra = dict(base_extra)
        pooled = _blend_pooled(extras, weights)
        if pooled is not None:
            out_extra["pooled_output"] = pooled
        return [[final, out_extra]]

    # -- mode 2: full-sequence weighted average (baseline) -----------
    def _full_average(self, clip, artists, weights, full_text, strength):
        tensors, extras = [], []
        for artist in artists:
            tensor, extra = _encode(clip, full_text(artist))
            tensors.append(tensor)
            extras.append(extra)

        max_len = max(t.shape[1] for t in tensors)
        blended = None
        norm_sum = None
        for tensor, w in zip(tensors, weights):
            pad = max_len - tensor.shape[1]
            if pad > 0:
                tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad))
            term = tensor * float(w)
            blended = term if blended is None else blended + term
            n = tensor.norm(dim=-1, keepdim=True) * float(w)
            norm_sum = n if norm_sum is None else norm_sum + n
        blended = _renormalise(blended, norm_sum) * float(strength)

        out_extra = dict(extras[0])
        pooled = _blend_pooled(extras, weights)
        if pooled is not None:
            out_extra["pooled_output"] = pooled
        return [[blended, out_extra]]


NODE_CLASS_MAPPINGS = {
    "AnimaArtistHiddenBlend": AnimaArtistHiddenBlend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaArtistHiddenBlend": "Anima Artist Blend (Qwen hidden states)",
}
