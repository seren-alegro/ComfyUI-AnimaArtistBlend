# Design notes

## Target

- Model family: Anima (2B / 2.9B), text encoder Qwen3 0.6B (`qwen_3_06b`).
- Environment: ComfyUI, encoding via
  `clip.encode_from_tokens_scheduled(clip.tokenize(text))`, the same
  path as stock CLIPTextEncode.

## Why prompt-string stacking fails

1. **Competition**: several rare proper-noun tokens in one sequence;
   attention spreads over them and the strongest/most-trained wins.
2. **Causality**: Qwen is a causal LM. A later artist tag's states are
   contextualised by the earlier ones; the earlier ones never see the
   later ones. The result depends on order, not on intended ratios.
3. **Distribution**: training captions effectively contain one artist
   tag, so multi-artist strings are out-of-distribution input.
4. **Not instruction-tuned**: Qwen3 0.6B *base* cannot be told in words
   to "mix A 70% / B 30%". The mixing must be done numerically, in the
   node, not phrased in the prompt.

## Why hidden states scale to 5+ artists

- Descriptor rewriting (expanding each artist into trait tags) grows
  the prompt with N, adds contradictory traits, and still competes in
  one pass.
- Hidden-state blending is O(N) cheap encoder passes (0.6B, short
  sequences — negligible against DiT sampling), the output length does
  not grow, and the weights stay meaningful at any N.

## Pooled-token construction

For artist at the end of the prompt, causality gives:

```
encode(base)            -> H_base            [L_b, D]
encode(base + artist_i) -> [H_base | H_art]  [L_b + L_i, D]
```

`H_base` is shared, so it is kept once. `H_art` has variable length
per artist, so it is mean-pooled to a single [1, D] vector — this also
absorbs tokeniser length differences between artist names. Blend:

```
w_i' ∝ w_i ^ sharpen ;  Σ w_i' = 1
v    = Σ w_i' · pool_i
‖v‖  := Σ w_i' · ‖pool_i‖        (norm restoration, anti-fade)
final = [ H_base ; v · strength ]
```

## Risks / open questions for the private test

- Linear midpoints in LLM hidden space may be off-manifold for the
  DiT cross-attention; if `pooled_token` underperforms, compare
  `full_average` and consider spherical interpolation.
- With 5+ artists the blend can collapse toward a generic mean style;
  `sharpen` is the first countermeasure.
- Tokeniser specifics (BOS/EOS handling in the installed ComfyUI
  build) affect the span cut; console logging of span sizes is the
  first diagnostic.
