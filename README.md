# ComfyUI-AnimaArtistBlend — a research log

**Status: research log, not a production tool.** This node was built to
answer one question experimentally:

> Can several artist styles be blended inside the hidden states of
> Anima's Qwen3 0.6B text encoder, instead of stacking artist tags in
> one prompt string?

Short answer, measured on a real pod and macstudio: **not by appending fabricated
tokens.** Pooled, amplified, per-artist and purity-boosted virtual
tokens all failed to produce a perceptible style blend, while two
approaches that stay inside real token sequences did get read — each
with its own catch. The measurements are the point of this repo.

## Test setup

- ComfyUI 0.44 (SwarmUI backend) on an RTX PRO 6000 MIG 24GB pod, mac studio m3 ultra 512gb
- Checkpoints: Anima base v1.0 and a 2.9B hybrid merge
- Fixed-seed controlled pairs throughout: same prompt, sampler, CFG
  and seed; only the node mode changed

## Findings

Artist vectors were measured against their shared component: in a
`base + artist` encoding, the artist-specific deviation is only
**20.4–22.2%** of the shared component — the pooled span is mostly a
summary of the whole prompt.

| Approach (version) | Result |
| --- | --- |
| `pooled_token` — blend spans into one appended virtual token (v0.1) | Under a fixed seed, outfit details changed; swapping one artist for another (nardack ↔ fuzichoco) produced nearly identical images |
| deviation gain ×3 on the pooled vectors (v0.2) | Deviation amplified, image still barely moved |
| `per_artist_tokens` — one virtual token per artist (v0.3) | Detail-level changes only |
| `extract_mode=isolated` — encode the tag alone (v0.4) | Deviation purity rose to **35.0%**; image still detail-level. Purity and token count were not the bottleneck |
| `full_average` — weighted average of whole real sequences (v0.1) | **First visible shift**: the whole tone went pale and soft — but that is the highest-weighted artist (cierra) winning, not a blend |
| `adapter_delta` — per-row perpendicular artist delta on the base sequence (v0.5) | Read by the DiT, but as a *re-voicing* of the base prompt (contrast/rendering shifts), not as imported artist identity. Measured delta: 14.9% of base norm, 95% of it perpendicular |

## Interpretation

A fabricated vector — pooled, averaged out of context, or extracted in
isolation — is off-format for the DiT: it resembles no token state the
model was trained to read. Interventions survive only where the
sequence stays close to real token states (`full_average`), or where
real base rows are bent in place (`adapter_delta`) — and neither
amounts to style blending. On top of that, a 0.6B encoder's knowledge
of low-resource artist tags is thin to begin with; some artists in the
test set have only a few hundred Danbooru images.

Projects that mix in cross-attention output space instead — see
Related work — sidestep the format problem entirely, at the cost of
patching the model and per-block work.

## The node

**Anima Artist Blend (Qwen hidden states)** (`conditioning/anima`)

- Inputs: a `CLIP` (Qwen3 0.6B from an Anima checkpoint/loader), a
  shared `base_prompt`, and up to **8 artist slots**, each with a
  weight. With no artist filled in, it behaves like a plain encode.
- Modes: `pooled_token`, `full_average`, `per_artist_tokens`,
  `adapter_delta`; `extract_mode` (`contextual` / `isolated`) applies
  to the pooled modes; `sharpen` exponentiates the weights before
  normalising; `deviation_gain` amplifies artist deviation in the
  pooled modes; `strength` scales the intervention.
- The node prints its span sizes and deviation measurements to the
  ComfyUI log (`[AnimaArtistBlend] ...`) — the numbers in the table
  above come from those lines.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/seren-alegro/ComfyUI-AnimaArtistBlend
# restart ComfyUI
```

No extra dependencies: PyTorch only, as provided by ComfyUI.

## Related work

- [An1X3R/Anima-Artist-Mixer](https://github.com/An1X3R/Anima-Artist-Mixer)
  — cross-attention output mixing plus an Adapter Mixer
  (perpendicular delta after the LLM adapter). The `adapter_delta`
  mode here follows the formula as published in that project's
  README; it is an independent implementation, no code copied.
  (That project is GPL-3.0 since its 26.8.1; this repo is MIT.)
- Localsmile's Anima Artist Cross-Attn Mix — per-block
  cross-attention output blending.
- n0va39's ComfyUI-EasyUseAnima — conditioning-space mixing modes
  including delta RMS.

## License

MIT — see `LICENSE`.
