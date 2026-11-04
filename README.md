# ComfyUI-AnimaArtistBlend

**Private test — not a public release.**

A ComfyUI custom node that blends multiple artist tags in the
hidden-state space of the Qwen3 0.6B text encoder (the Anima text
encoder), instead of stacking them in one prompt string.

## Why

Writing `artist_a, artist_b, artist_c, ...` in one prompt does not mix
the styles: the tags compete inside a single encoding pass and usually
one artist dominates. With a small causal-LM encoder this is worse —
later tokens see earlier ones but not the reverse — and training
captions almost never contained several artist tags at once.

## What the node does

Node: **Anima Artist Blend (Qwen hidden states)**
(`conditioning/anima`)

- Inputs: a `CLIP` (Qwen3 0.6B from an Anima checkpoint/loader), a
  shared `base_prompt`, and up to **8 artist slots**, each with a weight.
- Each artist is encoded **separately** as `base_prompt + artist_i`.
- `pooled_token` mode (default): the shared base prefix is taken once;
  each artist's trailing token span is mean-pooled to one vector; the
  vectors are blended and appended as a single *virtual artist token*.
- `full_average` mode: norm-restored weighted average of the whole
  sequences — the classic conditioning average, kept as an A/B baseline.
- `sharpen` (default 1.0): exponent applied to the weights before
  normalising. Above 1.0 it counteracts the "average grey style"
  collapse when blending 5+ artists.
- `strength` (default 1.0): scales the blended artist token.

With no artist filled in, the node behaves like a plain text encode.

## Install (test)

```bash
cd ComfyUI/custom_nodes
git clone <this-private-repo> ComfyUI-AnimaArtistBlend
# restart ComfyUI
```

No extra dependencies: PyTorch only, as provided by ComfyUI.

## Test plan

Same seed / prompt / sampler throughout:

1. one artist alone
2. 2 and 5 artists in a single prompt string (stock encode)
3. this node, `pooled_token`
4. this node, `full_average`

Judge whether (3) shows traits of several artists at once (palette of
A, eyes/line of B, …) instead of collapsing to one, and whether the
5-artist case stays distinct rather than turning into a generic mean.

First run: check the console lines
`[AnimaArtistBlend] '<artist>': span tokens = N` — if a span is 0 or
absurdly large, the base-length assumption needs adjusting for that
tokeniser build; report the numbers before changing anything.

## Known limitations (v0.1)

- Artist is always appended at the **end** of the base prompt
  (required by the causal-prefix argument above).
- EOS, if the tokeniser appends one, is included in the pooled span.
- Blending is linear in hidden space; whether the midpoint between
  artists is in-distribution for the DiT is exactly what the private
  test is meant to find out.

See `docs/DESIGN.md` for the full reasoning.
