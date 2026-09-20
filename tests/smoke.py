"""GPU-free smoke test for ComfyUI-H3-Loop.

Fakes ComfyUI's torch / node_helpers and a stub ComfyUI-H3-Project-Suite
module (so `_ensure_suite_patches` finds one), then drives the two nodes with
list-form fake AV latents `{"samples": [video_T, audio_T]}` exactly as a graph
would. Asserts:

  * H3LoopClose tail keyframe indices == (frame_count - covered) + step_offsets
  * trim_frames == the overlap window
  * an off-grid overlap (10) snaps DOWN to 5
  * seed_head copies the start slice into the head and attaches a noise mask
  * H3LoopTrim edge='tail' drops the trailing frames (keeps the head)

Run:  ~/ComfyUI/venv/bin/python tests/smoke.py   (or plain python3 -- numpy only)
"""

import importlib.util
import os
import sys
import types

import numpy as np

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_TESTS_DIR)


class T:
    """Minimal numpy-backed tensor stand-in."""

    def __init__(self, a):
        self.a = np.asarray(a)

    @property
    def shape(self):
        return self.a.shape

    @property
    def ndim(self):
        return self.a.ndim

    def __getitem__(self, idx):
        return T(self.a[idx])

    def __setitem__(self, idx, value):
        self.a[idx] = value.a if isinstance(value, T) else value

    def unsqueeze(self, d):
        return T(np.expand_dims(self.a, d))

    def clone(self):
        return T(self.a.copy())


class Nested:
    """Stand-in for ComfyUI's NestedTensor AV pair."""

    def __init__(self, parts):
        self.parts = list(parts)

    def unbind(self):
        return list(self.parts)


def _make_torch():
    t = types.ModuleType("torch")
    t.float32 = np.float32

    def _ones(shape, dtype=np.float32):
        return T(np.ones(shape, dtype=np.float32))
    t.ones = _ones

    nn = types.ModuleType("torch.nn")
    fn = types.ModuleType("torch.nn.functional")

    def _pad(x, pad):
        left, right = pad
        return T(np.pad(x.a, [(0, 0)] * (x.a.ndim - 1) + [(left, right)]))
    fn.pad = _pad
    nn.functional = fn
    t.nn = nn
    return t


def main():
    # --- fake the modules nodes.py imports at load time ---
    sys.modules["torch"] = _make_torch()

    captured = {}
    nh = types.ModuleType("node_helpers")

    def conditioning_set_values(cond, values, append=False):
        out = []
        for item in cond:
            meta = item[1].copy()
            for key, incoming in values.items():
                value = incoming
                if append and meta.get(key) is not None:
                    value = meta[key] + incoming
                meta[key] = value
            out.append([item[0], meta])
        captured.clear()
        if out:
            captured.update(out[0][1])
        return out
    nh.conditioning_set_values = conditioning_set_values
    sys.modules["node_helpers"] = nh

    # --- stub suite module so _ensure_suite_patches() finds an activator ---
    suite = types.ModuleType("fake_suite_nodes")
    suite.__file__ = "/x/custom_nodes/ComfyUI-H3-Project-Suite/nodes.py"
    activated = {"count": 0}
    suite._activate_inline_patches = lambda: activated.__setitem__(
        "count", activated["count"] + 1)
    sys.modules["fake_suite_nodes"] = suite

    # --- load the pack's nodes.py directly (dir name has hyphens) ---
    spec = importlib.util.spec_from_file_location(
        "h3loop_nodes", os.path.join(_PKG_DIR, "nodes.py"))
    nodes = importlib.util.module_from_spec(spec)
    sys.modules["h3loop_nodes"] = nodes
    spec.loader.exec_module(nodes)

    h, w = 480 // 16, 864 // 16
    C, M_steps, audio_t = 16, 37, 207
    frame_count = nodes._pixel_frames(M_steps)          # 124
    assert frame_count == 124, frame_count

    # ctx video: mark each step so slices are identifiable
    ctxA = T(np.zeros((1, C, M_steps, h, w), dtype=np.float32))
    for j in range(M_steps):
        ctxA[0, 0, j] = j
    loop_ctx = {"samples": [ctxA,
                            T(np.zeros((1, 32, 2, audio_t), dtype=np.float32))]}
    loop_tgt = {"samples": [
        T(np.zeros((1, C, M_steps, h, w), dtype=np.float32)),
        T(np.zeros((1, 32, 2, audio_t), dtype=np.float32))]}

    # --- H3LoopClose: tail keyframe indices + trim, no seeding ---
    lc = nodes.H3LoopClose()
    condL, trimL, outL = lc.apply(
        conditioning=[["c", {}]], latent=loop_tgt, context_latent=loop_ctx,
        overlap_length=22, seed_head=False, pin_head_keyframes=False)
    assert activated["count"] == 1, "suite patch activator must be called"
    covered = nodes._pixel_frames(7)                    # 22
    assert covered == 22, covered
    tail_p = [(frame_count - covered) + o for o in nodes._step_offsets(7)]
    got_p = sorted(kf[nodes.MC_KEY] for kf in captured["minimax_keyframes"])
    assert got_p == tail_p, (got_p, tail_p)
    assert trimL == covered == 22, trimL
    assert captured["minimax_frame_count"] == frame_count
    assert outL is loop_tgt, "seed_head=False returns the latent untouched"
    assert all(kf["resolved_frame_index"] == 0
               for kf in captured["minimax_keyframes"])
    print("H3LoopClose: tail keyframes at %s, trim %d" % (tail_p, trimL))

    # --- NEW: merge upstream head keyframes (chain-loop path) ---
    # Pre-seed the incoming conditioning with two "head" continue-pins, as an
    # upstream H3Context would leave them (MC_KEY 0 and 1). H3LoopClose must
    # UNION its 7 tail loop-pins with these two instead of clobbering them, and
    # the result must come out sorted by MC_KEY ascending: heads first, then the
    # tail window at [frame_count-covered ..].
    head_kfs = [
        {"resolved_frame_index": 0, nodes.MC_KEY: 0,
         "latent": T(np.zeros((1, C, 1, h, w), dtype=np.float32))},
        {"resolved_frame_index": 0, nodes.MC_KEY: 1,
         "latent": T(np.zeros((1, C, 1, h, w), dtype=np.float32))},
    ]
    seeded_cond = [["c", {"minimax_keyframes": head_kfs,
                          "minimax_frame_count": frame_count}]]
    _, trimM, _ = nodes.H3LoopClose().apply(
        conditioning=seeded_cond, latent=loop_tgt, context_latent=loop_ctx,
        overlap_length=22, seed_head=False, pin_head_keyframes=False)
    merged_idx = [kf[nodes.MC_KEY] for kf in captured["minimax_keyframes"]]
    assert len(merged_idx) == 9, (len(merged_idx), merged_idx)   # 2 head + 7 tail
    assert merged_idx == sorted(merged_idx), ("MC_KEY must be ascending",
                                              merged_idx)
    assert merged_idx[:2] == [0, 1], merged_idx                  # heads first
    assert merged_idx[2:] == tail_p, (merged_idx[2:], tail_p)    # then the tail
    assert trimM == 22, trimM
    assert captured["minimax_frame_count"] == frame_count
    print("H3LoopClose: merged 2 head + 7 tail keyframes -> %s, trim %d"
          % (merged_idx, trimM))

    # --- off-grid overlap 10 snaps DOWN to 5 ---
    cl3 = {"samples": [T(np.zeros((1, C, M_steps, h, w), dtype=np.float32)),
                       T(np.zeros((1, 32, 2, audio_t), dtype=np.float32))]}
    tg3 = {"samples": [T(np.zeros((1, C, M_steps, h, w), dtype=np.float32)),
                       T(np.zeros((1, 32, 2, audio_t), dtype=np.float32))]}
    _, trim_og, _ = nodes.H3LoopClose().apply(
        conditioning=[["c", {}]], latent=tg3, context_latent=cl3,
        overlap_length=10, seed_head=False)
    assert trim_og == 5, trim_og
    print("H3LoopClose: off-grid overlap 10 snapped to %d" % trim_og)

    # --- seed_head copies the start slice into the head + attaches a mask ---
    seed_ctx = {"samples": Nested([ctxA,
                T(np.zeros((1, 32, 2, audio_t), dtype=np.float32))])}
    seed_tgt = {"samples": Nested([
        T(np.zeros((1, C, M_steps, h, w), dtype=np.float32)),
        T(np.zeros((1, 32, 2, audio_t), dtype=np.float32))])}
    _, _, seeded = nodes.H3LoopClose().apply(
        conditioning=[["c", {}]], latent=seed_tgt, context_latent=seed_ctx,
        overlap_length=22, seed_head=True, head_hold=1.0)
    assert "noise_mask" in seeded, "seed_head must attach a noise mask"
    sv = seeded["samples"].unbind()[0]
    # first 7 head steps now equal the ctx's first 7 steps (channel 0 marks)
    assert float(sv.a[0, 0, 0, 0, 0]) == 0.0
    assert float(sv.a[0, 0, 6, 0, 0]) == 6.0, float(sv.a[0, 0, 6, 0, 0])
    mask = seeded["noise_mask"]
    assert float(mask.a[0, 0, 0, 0, 0]) == 0.0, "head_hold 1.0 => mask 0 (held)"
    assert float(mask.a[0, 0, 10, 0, 0]) == 1.0, "body denoises freely"
    print("seed_head: head steps copied, mask holds them")

    # --- NEW: seed_tail hard-holds clip 1's start in the TAIL, merging with a
    # pre-existing head-hold mask (as an upstream H3Context leaves) and NOT
    # clobbering it. Also exercises the keyframe-merge path (pre-seeded head
    # keyframes on the incoming conditioning). T=37, steps=7 -> tail [30:37],
    # (T-steps)=30 is a multiple of 5, so the phase lines up and the write runs.
    tail_head_mask = T(np.ones((1, 1, M_steps, h, w), dtype=np.float32))
    tail_head_mask[0, 0, 0:7] = 0.0            # H3Context's head-hold region
    seed_tail_ctx = {"samples": Nested([ctxA,
                     T(np.zeros((1, 32, 2, audio_t), dtype=np.float32))])}
    seed_tail_tgt = {
        "samples": Nested([
            T(np.zeros((1, C, M_steps, h, w), dtype=np.float32)),
            T(np.zeros((1, 32, 2, audio_t), dtype=np.float32))]),
        "noise_mask": tail_head_mask,
    }
    seeded_head_kfs = [
        {"resolved_frame_index": 0, nodes.MC_KEY: 0,
         "latent": T(np.zeros((1, C, 1, h, w), dtype=np.float32))},
        {"resolved_frame_index": 0, nodes.MC_KEY: 1,
         "latent": T(np.zeros((1, C, 1, h, w), dtype=np.float32))},
    ]
    tail_cond_in = [["c", {"minimax_keyframes": seeded_head_kfs,
                           "minimax_frame_count": frame_count}]]
    _, trimT, seededT = nodes.H3LoopClose().apply(
        conditioning=tail_cond_in, latent=seed_tail_tgt,
        context_latent=seed_tail_ctx, overlap_length=22,
        seed_head=False, seed_tail=True, tail_hold=0.6)
    # (d) phase assertion holds for T=37 / steps=7
    assert (M_steps - 7) % 5 == 0, "phase: (T-steps) must be a multiple of 5"
    svT = seededT["samples"].unbind()[0]
    # (a) tail steps [T-7:T] equal ctx head steps [0:7] (channel 0 marks step j)
    for j in range(7):
        got = float(svT.a[0, 0, (M_steps - 7) + j, 0, 0])
        assert got == float(j), ("tail step mismatch", j, got)
    assert float(svT.a[0, 0, 0, 0, 0]) == 0.0, "head of latent untouched by tail"
    maskT = seededT["noise_mask"]
    # (b) noise_mask tail region == 1 - tail_hold
    assert abs(float(maskT.a[0, 0, M_steps - 7, 0, 0]) - 0.4) < 1e-6, \
        float(maskT.a[0, 0, M_steps - 7, 0, 0])
    assert abs(float(maskT.a[0, 0, M_steps - 1, 0, 0]) - 0.4) < 1e-6
    # (c) the pre-existing head-hold mask region is preserved (still 0.0 held)
    assert float(maskT.a[0, 0, 0, 0, 0]) == 0.0, "head-hold must survive seed_tail"
    assert float(maskT.a[0, 0, 6, 0, 0]) == 0.0, "head-hold must survive seed_tail"
    # a mid region neither head nor tail stays free to denoise
    assert float(maskT.a[0, 0, 20, 0, 0]) == 1.0, "body denoises freely"
    assert trimT == 22, trimT
    print("seed_tail: tail steps hard-held from clip 1 start, head mask "
          "preserved, phase OK")

    # --- seed_tail phase-safety: off-grid clip length SKIPS the tail write and
    # leaves the latent uncorrupted (falls back to cond-rows only). T=38,
    # steps=7 -> (T-steps)=31, not a multiple of 5.
    og_steps = 38
    skip_ctx = {"samples": Nested([
        T(np.zeros((1, C, og_steps, h, w), dtype=np.float32)),
        T(np.zeros((1, 32, 2, audio_t), dtype=np.float32))])}
    skip_tgt = {"samples": Nested([
        T(np.zeros((1, C, og_steps, h, w), dtype=np.float32)),
        T(np.zeros((1, 32, 2, audio_t), dtype=np.float32))])}
    assert (og_steps - 7) % 5 != 0, "test setup: 38 must be off-grid"
    _, _, seededSkip = nodes.H3LoopClose().apply(
        conditioning=[["c", {}]], latent=skip_tgt, context_latent=skip_ctx,
        overlap_length=22, seed_head=False, seed_tail=True, tail_hold=1.0)
    assert seededSkip is skip_tgt, "off-grid seed_tail must return latent as-is"
    assert "noise_mask" not in seededSkip, "off-grid seed_tail must not touch latent"
    print("seed_tail: off-grid length skips the tail write (latent uncorrupted)")

    # --- H3LoopTrim edge='tail' drops the trailing frames, keeps the head ---
    imgs = T(np.arange(124 * 2 * 2 * 3, dtype=np.float32).reshape(124, 2, 2, 3))
    outT, audioT = nodes.H3LoopTrim().trim(
        images=imgs, trim_frames=22, edge="tail")
    assert int(outT.shape[0]) == 102, outT.shape
    assert float(outT.a[0, 0, 0, 0]) == float(imgs.a[0, 0, 0, 0])
    assert float(outT.a[101, 0, 0, 0]) == float(imgs.a[101, 0, 0, 0])
    assert audioT is None
    # default edge is 'tail'
    outD, _ = nodes.H3LoopTrim().trim(images=imgs, trim_frames=22)
    assert float(outD.a[101, 0, 0, 0]) == float(imgs.a[101, 0, 0, 0])
    print("H3LoopTrim: edge=tail drops last 22 frames (kept head), "
          "default edge is tail")

    # --- missing suite => clear RuntimeError ---
    del sys.modules["fake_suite_nodes"]
    try:
        nodes.H3LoopClose().apply(
            conditioning=[["c", {}]], latent=loop_tgt, context_latent=loop_ctx,
            overlap_length=22, seed_head=False)
        raise AssertionError("expected RuntimeError without the suite loaded")
    except RuntimeError as exc:
        assert "ComfyUI-H3-Project-Suite" in str(exc), str(exc)
    print("H3LoopClose: missing suite raises a clear RuntimeError")

    print("smoke test passed")


def test_parse_prompts_shared():
    import h3loop_nodes as N
    assert N._parse_prompts("hello", 3) == ["hello","hello","hello"]


def test_parse_prompts_per_clip():
    import h3loop_nodes as N
    s = "a\n===\nb\n===\nc"
    assert N._parse_prompts(s, 3) == ["a","b","c"]


def test_parse_prompts_mismatch_raises():
    import h3loop_nodes as N
    try:
        N._parse_prompts("a\n===\nb", 3); assert False
    except ValueError as e:
        assert "1 (shared) or exactly" in str(e)


def _chainloop_kwargs(**overrides):
    kwargs = dict(
        model=None, clip=None, vae=None, ref_image_0=None, prompts="x",
        num_clips=1, length=124, overlap=22, width=768, height=1344,
        steps=8, sampler_name="er_sde", scheduler="beta", denoise=1.0,
        seed=0, seed_tail=True, head_hold=1.0, tail_hold=1.0,
    )
    kwargs.update(overrides)
    return kwargs


def test_h3chainloop_registered():
    import h3loop_nodes as N
    assert "H3ChainLoop" in N.NODE_CLASS_MAPPINGS
    assert "H3ChainLoop" in N.NODE_DISPLAY_NAME_MAPPINGS


def test_h3chainloop_width_guard():
    import h3loop_nodes as N
    try:
        N.H3ChainLoop().run(**_chainloop_kwargs(width=700))
        assert False, "expected ValueError"
    except ValueError as e:
        assert "divisible by 32" in str(e), str(e)


def test_h3chainloop_overlap_not_in_context_lengths_guard():
    import h3loop_nodes as N
    try:
        N.H3ChainLoop().run(**_chainloop_kwargs(overlap=17))
        assert False, "expected ValueError"
    except ValueError as e:
        assert "must be one of" in str(e), str(e)


def test_h3chainloop_overlap_ge_length_guard():
    import h3loop_nodes as N
    # overlap=200 with length=124: not in CONTEXT_LENGTHS *and* >= length,
    # so this hits the "must be one of" guard first (guard order); still a
    # ValueError, per the task spec.
    try:
        N.H3ChainLoop().run(**_chainloop_kwargs(overlap=200, length=124))
        assert False, "expected ValueError"
    except ValueError:
        pass
    # Isolate the overlap>=length guard specifically: overlap=56 IS a valid
    # CONTEXT_LENGTHS entry, so this can only fail the third guard.
    try:
        N.H3ChainLoop().run(**_chainloop_kwargs(overlap=56, length=39))
        assert False, "expected ValueError"
    except ValueError as e:
        assert "must be < length" in str(e), str(e)


def test_h3chainloop_2overlap_ge_length_guard_chain():
    import h3loop_nodes as N
    # num_clips>=2 with 2*overlap >= length should fail: the final clip
    # is trimmed on both ends (overlap frames on head + overlap on tail).
    # length=56, overlap=39 => 2*39=78 > 56, so this violates the chain guard.
    try:
        N.H3ChainLoop().run(**_chainloop_kwargs(
            num_clips=2, length=56, overlap=39))
        assert False, "expected ValueError"
    except ValueError as e:
        assert "2*overlap" in str(e), str(e)


def test_h3chainloop_2overlap_ge_length_guard_num_clips_1():
    import h3loop_nodes as N
    # num_clips=1 should NOT trigger the 2*overlap guard, only the
    # overlap >= length guard. With length=56 and overlap=39, overlap < length
    # but 2*overlap >= length. num_clips=1 tail-trims once, not twice,
    # so it should pass the guards and proceed to _gen_clip.
    # Since we're mocking/testing the guards only, we expect a different error
    # from _gen_clip's execution (not the 2*overlap guard message).
    try:
        N.H3ChainLoop().run(**_chainloop_kwargs(
            num_clips=1, length=56, overlap=39))
        # If it gets past guards without raising, that's OK (it'll fail
        # later in _gen_clip internals, which is expected with fake mocks).
        # The key is that the "2*overlap" message must NOT appear.
    except (ValueError, RuntimeError) as e:
        # If an exception is raised, it must NOT be from the 2*overlap guard.
        assert "2*overlap" not in str(e), \
            "num_clips=1 must not trigger the 2*overlap guard: " + str(e)


def test_clamp_length():
    import h3loop_nodes as N
    assert N._clamp_length(2) == 5
    assert N._clamp_length(5000) == 3600
    assert N._clamp_length(124) == 124


def test_ref_bundle():
    import h3loop_nodes as N
    # single ref -> slot 0
    assert N._ref_bundle(["a"]) == {"ref_image_0": "a"}
    # three contiguous refs -> slots 0,1,2
    assert N._ref_bundle(["a", "b", "c"]) == {
        "ref_image_0": "a", "ref_image_1": "b", "ref_image_2": "c"}
    # gaps (unwired middle slot) are skipped but slot indices preserved
    assert N._ref_bundle(["a", None, "c"]) == {
        "ref_image_0": "a", "ref_image_2": "c"}
    # all 9 slots
    full = ["r%d" % i for i in range(9)]
    b = N._ref_bundle(full)
    assert len(b) == 9 and b["ref_image_0"] == "r0" and b["ref_image_8"] == "r8"
    # empty / all-None -> empty bundle
    assert N._ref_bundle([None, None]) == {}
    print("_ref_bundle: 1/3/gap/9/empty refs -> correct slot-keyed bundle")


# --------------------------------------------------------------------------
# Task 4: _gen_clip -- mocked (no GPU, no real render). Fakes
# MiniMaxH3ReferenceToVideo/H3Context/H3LoopClose via h3loop_nodes's own
# NODE_CLASS_MAPPINGS (that's what _cls checks first, see nodes.py) and
# monkeypatches h3loop_nodes._sample wholesale, so nothing here touches
# comfy.* at all.
# --------------------------------------------------------------------------

def _fake_ctx(**overrides):
    kwargs = dict(
        model="MODEL", clip="CLIP", vae="VAE", ref_images=["REFIMG"],
        width=768, height=1344, length=124, overlap=22, steps=8,
        sampler_name="er_sde", scheduler="beta", denoise=1.0,
        seed_tail=True, head_hold=1.0, tail_hold=1.0,
    )
    kwargs.update(overrides)
    return types.SimpleNamespace(**kwargs)


def _install_gen_clip_fakes():
    import h3loop_nodes as N

    ref_calls = []

    class FakeRef2VA:
        @classmethod
        def execute(cls, **kwargs):
            ref_calls.append(kwargs)
            return ("COND0", "LATENT0")

    ctx_calls = []

    class FakeH3Context:
        def apply(self, **kwargs):
            ctx_calls.append(kwargs)
            return ("COND_CTX", 5, "LATENT_CTX")

    loop_calls = []

    class FakeH3LoopClose:
        def apply(self, **kwargs):
            loop_calls.append(kwargs)
            return ("COND_LOOP", 7, "LATENT_LOOP")

    sample_calls = []

    def fake_sample(model, cond, latent, sampler_name, scheduler, steps,
                     denoise, seed):
        sample_calls.append(dict(
            model=model, cond=cond, latent=latent, sampler_name=sampler_name,
            scheduler=scheduler, steps=steps, denoise=denoise, seed=seed))
        return "SAMPLED"

    originals = {
        name: N.NODE_CLASS_MAPPINGS.get(name)
        for name in ("MiniMaxH3ReferenceToVideo", "H3Context", "H3LoopClose")
    }
    orig_sample = N._sample
    N.NODE_CLASS_MAPPINGS["MiniMaxH3ReferenceToVideo"] = FakeRef2VA
    N.NODE_CLASS_MAPPINGS["H3Context"] = FakeH3Context
    N.NODE_CLASS_MAPPINGS["H3LoopClose"] = FakeH3LoopClose
    N._sample = fake_sample

    return dict(ref_calls=ref_calls, ctx_calls=ctx_calls, loop_calls=loop_calls,
                sample_calls=sample_calls, originals=originals,
                orig_sample=orig_sample)


def _restore_gen_clip_fakes(state):
    import h3loop_nodes as N
    for name, val in state["originals"].items():
        if val is None:
            N.NODE_CLASS_MAPPINGS.pop(name, None)
        else:
            N.NODE_CLASS_MAPPINGS[name] = val
    N._sample = state["orig_sample"]


def test_gen_clip_plain():
    """No context_latent, no loop_target_latent: Ref2VA -> sample only."""
    import h3loop_nodes as N
    state = _install_gen_clip_fakes()
    try:
        node = N.H3ChainLoop()
        ctx = _fake_ctx()
        out_latent, trim_head, trim_tail = node._gen_clip(
            ctx, "prompt A", context_latent=None, loop_target_latent=None,
            seed=123)

        assert len(state["ref_calls"]) == 1, state["ref_calls"]
        rc = state["ref_calls"][0]
        assert rc["prompt"] == "prompt A"
        assert rc["clip"] == "CLIP" and rc["vae"] == "VAE"
        assert rc["width"] == 768 and rc["height"] == 1344 and rc["length"] == 124
        assert rc["ref_images"] == {"ref_image_0": "REFIMG"}

        assert len(state["ctx_calls"]) == 0, "H3Context must not run without context_latent"
        assert len(state["loop_calls"]) == 0, "H3LoopClose must not run without loop_target_latent"

        assert len(state["sample_calls"]) == 1
        sc = state["sample_calls"][0]
        assert sc["cond"] == "COND0" and sc["latent"] == "LATENT0"
        assert sc["model"] == "MODEL" and sc["seed"] == 123
        assert sc["sampler_name"] == "er_sde" and sc["scheduler"] == "beta"
        assert sc["steps"] == 8 and sc["denoise"] == 1.0

        assert out_latent == "SAMPLED"
        assert trim_head == 0
        assert trim_tail == 0
    finally:
        _restore_gen_clip_fakes(state)
    print("_gen_clip: plain clip (no context, no loop) samples once, trims 0/0")


def test_gen_clip_multiref():
    """Multiple wired references (with a gap) all reach Ref2VA, slot-keyed."""
    import h3loop_nodes as N
    state = _install_gen_clip_fakes()
    try:
        ctx = _fake_ctx(ref_images=["A", None, "C"])
        N.H3ChainLoop()._gen_clip(
            ctx, "p", context_latent=None, loop_target_latent=None, seed=1)
        assert state["ref_calls"][0]["ref_images"] == {
            "ref_image_0": "A", "ref_image_2": "C"}, state["ref_calls"][0]
    finally:
        _restore_gen_clip_fakes(state)
    print("_gen_clip: multi-ref bundle passes all wired slots, skips gaps")


def test_gen_clip_context_only():
    """context_latent given, no loop close: H3Context runs, seed_head=True."""
    import h3loop_nodes as N
    state = _install_gen_clip_fakes()
    try:
        node = N.H3ChainLoop()
        ctx = _fake_ctx()
        out_latent, trim_head, trim_tail = node._gen_clip(
            ctx, "prompt B", context_latent="CTXLAT", loop_target_latent=None,
            seed=7)

        assert len(state["ctx_calls"]) == 1, state["ctx_calls"]
        c = state["ctx_calls"][0]
        assert c["conditioning"] == "COND0" and c["latent"] == "LATENT0"
        assert c["context_latent"] == "CTXLAT"
        assert c["anchor_mode"] == "head"
        assert c["seed_head"] is True
        assert c["encode_mode"] == "video"
        assert c["crop"] == "disabled"
        assert c["video_source"] == "latent"
        assert c["context_length"] == ctx.overlap
        assert c["head_hold"] == ctx.head_hold

        assert len(state["loop_calls"]) == 0, "H3LoopClose must not run without loop_target_latent"

        assert len(state["sample_calls"]) == 1
        sc = state["sample_calls"][0]
        assert sc["cond"] == "COND_CTX" and sc["latent"] == "LATENT_CTX"

        assert out_latent == "SAMPLED"
        assert trim_head == 5
        assert trim_tail == 0
    finally:
        _restore_gen_clip_fakes(state)
    print("_gen_clip: context-only clip runs H3Context (anchor_mode=head, "
          "seed_head=True), trims 5/0")


def test_gen_clip_context_and_loop():
    """Both context_latent and loop_target_latent given (final chain clip):
    H3Context then H3LoopClose chain, seed_head=False on the loop close
    (an upstream continue-pin already seeded the head)."""
    import h3loop_nodes as N
    state = _install_gen_clip_fakes()
    try:
        node = N.H3ChainLoop()
        ctx = _fake_ctx()
        out_latent, trim_head, trim_tail = node._gen_clip(
            ctx, "prompt C", context_latent="CTXLAT",
            loop_target_latent="LOOPLAT", seed=9)

        assert len(state["ctx_calls"]) == 1
        assert len(state["loop_calls"]) == 1
        lc = state["loop_calls"][0]
        assert lc["conditioning"] == "COND_CTX" and lc["latent"] == "LATENT_CTX", (
            "H3LoopClose must chain off H3Context's output, not Ref2VA's raw output")
        assert lc["context_latent"] == "LOOPLAT"
        assert lc["overlap_length"] == ctx.overlap
        assert lc["seed_head"] is False, "seed_head must be False when context_latent was given"
        assert lc["seed_tail"] == ctx.seed_tail
        assert lc["merge_context"] is True
        assert lc["tail_hold"] == ctx.tail_hold
        assert lc["head_hold"] == ctx.head_hold
        assert lc["pin_head_keyframes"] is False

        assert len(state["sample_calls"]) == 1
        sc = state["sample_calls"][0]
        assert sc["cond"] == "COND_LOOP" and sc["latent"] == "LATENT_LOOP"

        assert out_latent == "SAMPLED"
        assert trim_head == 5
        assert trim_tail == 7
    finally:
        _restore_gen_clip_fakes(state)
    print("_gen_clip: context+loop clip chains H3Context -> H3LoopClose "
          "(seed_head=False), trims 5/7")


def test_gen_clip_loop_only():
    """N=1 harvest-then-loop path: loop_target_latent given but
    context_latent is None (first/only clip closing its own loop) ->
    H3LoopClose gets seed_head=True."""
    import h3loop_nodes as N
    state = _install_gen_clip_fakes()
    try:
        node = N.H3ChainLoop()
        ctx = _fake_ctx()
        out_latent, trim_head, trim_tail = node._gen_clip(
            ctx, "prompt D", context_latent=None, loop_target_latent="LOOPLAT",
            seed=3)

        assert len(state["ctx_calls"]) == 0, "H3Context must not run without context_latent"
        assert len(state["loop_calls"]) == 1
        lc = state["loop_calls"][0]
        assert lc["conditioning"] == "COND0" and lc["latent"] == "LATENT0", (
            "H3LoopClose must chain off Ref2VA's raw output when there was no context pin")
        assert lc["context_latent"] == "LOOPLAT"
        assert lc["seed_head"] is True, "seed_head must be True when context_latent is None"
        assert lc["seed_tail"] == ctx.seed_tail

        assert len(state["sample_calls"]) == 1
        sc = state["sample_calls"][0]
        assert sc["cond"] == "COND_LOOP" and sc["latent"] == "LATENT_LOOP"

        assert out_latent == "SAMPLED"
        assert trim_head == 0
        assert trim_tail == 7
    finally:
        _restore_gen_clip_fakes(state)
    print("_gen_clip: loop-only clip (N=1 harvest->loop) runs H3LoopClose "
          "with seed_head=True, trims 0/7")


# --------------------------------------------------------------------------
# Task 5: H3ChainLoop.run -- mocked (no GPU). `_gen_clip` itself is faked
# (its own internals are covered by Task 4's tests above), decode is faked
# with a tiny tensor tagged (decode_call_index*10000 + local_frame_index) so
# real H3LoopTrim slicing + real torch.cat concat order/direction are all
# actually exercised, not just assumed from mocked call args.
# --------------------------------------------------------------------------

def _install_run_fakes(length):
    import h3loop_nodes as N

    gen_calls = []
    counter = {"n": 0}

    def fake_gen_clip(self, ctx, prompt, context_latent, loop_target_latent, seed):
        counter["n"] += 1
        sentinel = "LATENT_%d" % counter["n"]
        trim_head = ctx.overlap if context_latent is not None else 0
        trim_tail = ctx.overlap if loop_target_latent is not None else 0
        gen_calls.append(dict(prompt=prompt, context_latent=context_latent,
                              loop_target_latent=loop_target_latent, seed=seed,
                              sentinel=sentinel))
        return sentinel, trim_head, trim_tail

    decode_calls = []

    class FakeVAEDecode:
        def decode(self, vae, samples):
            decode_calls.append(samples)
            idx = len(decode_calls)
            arr = np.zeros((length, 2, 2, 3), dtype=np.float32)
            arr[:, 0, 0, 0] = idx * 10000 + np.arange(length)
            return (T(arr),)

    orig_gen_clip = N.H3ChainLoop._gen_clip
    orig_vae_decode = N.NODE_CLASS_MAPPINGS.get("VAEDecode")
    N.H3ChainLoop._gen_clip = fake_gen_clip
    N.NODE_CLASS_MAPPINGS["VAEDecode"] = FakeVAEDecode

    return dict(gen_calls=gen_calls, decode_calls=decode_calls,
                orig_gen_clip=orig_gen_clip, orig_vae_decode=orig_vae_decode)


def _restore_run_fakes(state):
    import h3loop_nodes as N
    N.H3ChainLoop._gen_clip = state["orig_gen_clip"]
    if state["orig_vae_decode"] is None:
        N.NODE_CLASS_MAPPINGS.pop("VAEDecode", None)
    else:
        N.NODE_CLASS_MAPPINGS["VAEDecode"] = state["orig_vae_decode"]


def test_h3chainloop_run_n1():
    """num_clips=1: two-pass single loop -- harvest then loop-against-harvest,
    decode only the loop pass, tail-trim only (no head trim)."""
    import h3loop_nodes as N
    length, overlap = 124, 22
    state = _install_run_fakes(length)
    # add torch.cat to the fake torch module in sys.modules (nodes.py's own
    # `torch` global is bound to that fake module object) so real
    # torch.cat(frames, dim=0) inside run() actually concatenates.
    sys.modules["torch"].cat = lambda tensors, dim=0: T(
        np.concatenate([t.a for t in tensors], axis=dim))
    try:
        node = N.H3ChainLoop()
        out, = node.run(**_chainloop_kwargs(
            num_clips=1, length=length, overlap=overlap, seed=100))

        calls = state["gen_calls"]
        assert len(calls) == 2, calls
        c1, c2 = calls
        assert c1["context_latent"] is None and c1["loop_target_latent"] is None
        assert c1["seed"] == 100
        assert c2["context_latent"] is None
        assert c2["loop_target_latent"] == c1["sentinel"], (
            "pass 2 must loop against pass 1's harvest latent")
        assert c2["seed"] == 101

        assert len(state["decode_calls"]) == 1, (
            "only the loop pass is decoded, not the harvest")
        assert state["decode_calls"][0] == c2["sentinel"]

        expected = length - overlap
        assert int(out.shape[0]) == expected, (out.shape, expected)
        # tail-trim only: kept frames are the FRONT of the raw range
        # (10000..10123), i.e. no head trim occurred.
        assert float(out.a[0, 0, 0, 0]) == 10000.0
        assert float(out.a[expected - 1, 0, 0, 0]) == 10000.0 + expected - 1
    finally:
        _restore_run_fakes(state)
        del sys.modules["torch"].cat
    print("H3ChainLoop.run: num_clips=1 harvest+loop, %d frames (tail-trimmed "
          "%d, no head trim)" % (expected, overlap))


def test_h3chainloop_run_n3():
    """num_clips=3 chain: clip1 kept whole, clip2 head-trimmed, clip3
    head+tail-trimmed (closes the loop back to clip1), concatenated in
    order with the right frame counts."""
    import h3loop_nodes as N
    length, overlap = 124, 22
    state = _install_run_fakes(length)
    sys.modules["torch"].cat = lambda tensors, dim=0: T(
        np.concatenate([t.a for t in tensors], axis=dim))
    try:
        node = N.H3ChainLoop()
        prompts = "p1\n===\np2\n===\np3"
        out, = node.run(**_chainloop_kwargs(
            num_clips=3, length=length, overlap=overlap, prompts=prompts,
            seed=100))

        calls = state["gen_calls"]
        assert len(calls) == 3, calls
        c1, c2, c3 = calls
        assert c1["prompt"] == "p1" and c2["prompt"] == "p2" and c3["prompt"] == "p3"
        assert c1["context_latent"] is None and c1["loop_target_latent"] is None
        assert c1["seed"] == 100
        assert c2["context_latent"] == c1["sentinel"]
        assert c2["loop_target_latent"] is None
        assert c2["seed"] == 101
        assert c3["context_latent"] == c2["sentinel"]
        assert c3["loop_target_latent"] == c1["sentinel"], (
            "only the LAST clip closes the loop, against clip 1's latent")
        assert c3["seed"] == 102

        assert len(state["decode_calls"]) == 3
        assert state["decode_calls"] == [c1["sentinel"], c2["sentinel"], c3["sentinel"]]

        clip1_n = length
        clip2_n = length - overlap
        clip3_n = length - 2 * overlap
        total = clip1_n + clip2_n + clip3_n
        assert total == 3 * (length - overlap), total
        assert int(out.shape[0]) == total, (out.shape, total)

        # clip1: kept whole, raw values 10000..10123
        assert float(out.a[0, 0, 0, 0]) == 10000.0
        assert float(out.a[clip1_n - 1, 0, 0, 0]) == 10000.0 + clip1_n - 1

        # clip2: head-trimmed by `overlap` -- first kept value is
        # 20000+overlap (the leading `overlap` frames were dropped), last
        # kept value is the untouched raw tail 20000+length-1.
        base2 = clip1_n
        assert float(out.a[base2, 0, 0, 0]) == 20000.0 + overlap
        assert float(out.a[base2 + clip2_n - 1, 0, 0, 0]) == 20000.0 + length - 1

        # clip3: head-trimmed by `overlap` AND tail-trimmed by `overlap` --
        # first kept value is 30000+overlap, last kept is 30000+length-1-overlap.
        base3 = clip1_n + clip2_n
        assert float(out.a[base3, 0, 0, 0]) == 30000.0 + overlap
        assert float(out.a[base3 + clip3_n - 1, 0, 0, 0]) == 30000.0 + length - 1 - overlap
    finally:
        _restore_run_fakes(state)
        del sys.modules["torch"].cat
    print("H3ChainLoop.run: num_clips=3 chain, frames %d+%d+%d=%d, "
          "concatenated in order" % (clip1_n, clip2_n, clip3_n, total))


if __name__ == "__main__":
    main()
    test_parse_prompts_shared()
    test_parse_prompts_per_clip()
    test_parse_prompts_mismatch_raises()
    print("All _parse_prompts tests passed")
    test_h3chainloop_registered()
    test_h3chainloop_width_guard()
    test_h3chainloop_overlap_not_in_context_lengths_guard()
    test_h3chainloop_overlap_ge_length_guard()
    test_h3chainloop_2overlap_ge_length_guard_chain()
    test_h3chainloop_2overlap_ge_length_guard_num_clips_1()
    test_clamp_length()
    test_ref_bundle()
    print("All H3ChainLoop shell/guard tests passed")
    test_gen_clip_plain()
    test_gen_clip_multiref()
    test_gen_clip_context_only()
    test_gen_clip_context_and_loop()
    test_gen_clip_loop_only()
    print("All _gen_clip tests passed")
    test_h3chainloop_run_n1()
    test_h3chainloop_run_n3()
    print("All H3ChainLoop.run orchestration tests passed")
