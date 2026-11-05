"""Unit tests — run inside a ComfyUI python env (needs torch):

    python -m pytest tests/ -q

The weight tests also pass without torch if it is stubbed, but the
tensor tests genuinely need it.
"""

import importlib.util
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
torch = pytest.importorskip("torch")

from nodes import AnimaArtistHiddenBlend, sharpened_weights  # noqa: E402


# ---------- weight maths (pure python logic) ----------

def test_weights_normalise():
    w = sharpened_weights([1.0, 1.0, 2.0], 1.0)
    assert w == pytest.approx([0.25, 0.25, 0.5])


def test_weights_sharpen():
    soft = sharpened_weights([2.0, 1.0], 1.0)
    sharp = sharpened_weights([2.0, 1.0], 2.0)
    assert sharp[0] > soft[0] > 0.5
    assert sum(sharp) == pytest.approx(1.0)


def test_weights_all_zero_falls_back_to_uniform():
    assert sharpened_weights([0.0, 0.0], 1.0) == [0.5, 0.5]


# ---------- mock CLIP ----------

class MockClip:
    """Deterministic fake encoder.

    tokenize: one token per word. encode: hidden state of token j is a
    vector filled with (position + hash-of-word), so the shared base
    prefix is identical across passes and artist spans are separable.
    """

    def tokenize(self, text):
        return [w for w in text.replace(",", " ").split() if w]

    def encode_from_tokens_scheduled(self, tokens):
        dim = 4
        rows = []
        for pos, tok in enumerate(tokens):
            v = float((sum(tok.encode()) % 7) + pos)
            rows.append([v] * dim)
        tensor = torch.tensor([rows], dtype=torch.float32)
        pooled = tensor.mean(dim=1)
        return [[tensor, {"pooled_output": pooled}]]


def _run(artists_weights, mode="pooled_token", sharpen=1.0, gain=1.0):
    node = AnimaArtistHiddenBlend()
    kwargs = {}
    for i in range(1, 9):
        if i <= len(artists_weights):
            kwargs[f"artist_{i}"], kwargs[f"weight_{i}"] = artists_weights[i - 1]
        else:
            kwargs[f"artist_{i}"], kwargs[f"weight_{i}"] = "", 1.0
    (result,) = node.blend(MockClip(), "1girl, solo", mode, sharpen, 1.0,
                           deviation_gain=gain, **kwargs)
    return result[0]


def test_pooled_token_appends_one_virtual_token():
    tensor, _ = _run([("artist_a", 1.0), ("artist_b", 1.0)])
    base_len = 2  # "1girl", "solo"
    assert tensor.shape[1] == base_len + 1


def test_pooled_token_base_prefix_unchanged():
    blended, _ = _run([("artist_a", 1.0), ("artist_b", 3.0)])
    base_only, _ = _run([])
    assert torch.allclose(blended[:, :2, :], base_only[:, :2, :])


def test_pooled_token_norm_restored():
    tensor, _ = _run([("artist_a", 1.0), ("artist_b", 1.0)])
    blended_norm = tensor[:, -1, :].norm(dim=-1).item()
    # individual artist pooled norms, recomputed through the mock
    clip = MockClip()
    norms = []
    for a in ("artist_a", "artist_b"):
        t, _ = clip.encode_from_tokens_scheduled(
            clip.tokenize("1girl, solo, " + a))[0]
        norms.append(t[:, 2:, :].mean(dim=1).norm(dim=-1).item())
    assert blended_norm == pytest.approx(sum(norms) / 2, rel=1e-5)


def test_full_average_shape_and_strength():
    tensor, _ = _run([("artist_a", 1.0), ("artist bb", 1.0)],
                     mode="full_average")
    assert tensor.shape[1] == 4  # longest sequence wins
    assert tensor.shape[2] == 4


def test_no_artist_is_plain_encode():
    tensor, _ = _run([])
    clip = MockClip()
    expected, _ = clip.encode_from_tokens_scheduled(
        clip.tokenize("1girl, solo"))[0]
    assert torch.equal(tensor, expected)


# ---------- pooled_output handling (pod regression, 2026-11-05) ----------

from nodes import _blend_pooled  # noqa: E402


def test_blend_pooled_none_values_are_absent():
    # Anima's Qwen3 TE returns pooled_output=None; the old code crashed
    # with AttributeError: 'NoneType' object has no attribute 'dim'.
    extras = [{"pooled_output": None}, {"pooled_output": None}]
    assert _blend_pooled(extras, [0.5, 0.5]) is None


def test_blend_pooled_mixed_none_skips():
    extras = [{"pooled_output": torch.ones(1, 4)}, {"pooled_output": None}]
    assert _blend_pooled(extras, [0.5, 0.5]) is None


def test_deviation_gain_separates_weightings():
    # Pod finding (2026-11-05): swapping one artist barely moved the
    # image. With deviation gain, two blends that differ only in which
    # artist dominates must land farther apart than at gain 1.
    a1, _ = _run([("artist_a", 3.0), ("artist_b", 1.0)], gain=1.0)
    b1, _ = _run([("artist_a", 1.0), ("artist_b", 3.0)], gain=1.0)
    a3, _ = _run([("artist_a", 3.0), ("artist_b", 1.0)], gain=3.0)
    b3, _ = _run([("artist_a", 1.0), ("artist_b", 3.0)], gain=3.0)
    d1 = (a1[:, -1, :] - b1[:, -1, :]).norm().item()
    d3 = (a3[:, -1, :] - b3[:, -1, :]).norm().item()
    assert d3 > d1
