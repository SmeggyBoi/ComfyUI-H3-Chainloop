"""H3 Loop: seamless-loop nodes for MiniMax H3, as a sidecar pack.

These two nodes were extracted from ComfyUI-H3-Project-Suite so that pack's
author can update it upstream without colliding with our loop work. They
live here, but they still RELY on the suite at runtime: the suite installs
the marker-gated H3 keyframe layout patch that lifts H3's first/last-only
anchor restriction, and our tail keyframes are interior anchors that only
build once that patch is active.

Why this module is import-time self-contained (no `from ...Project-Suite`):
custom_nodes load ALPHABETICALLY, so `ComfyUI-H3-Loop` imports BEFORE
`ComfyUI-H3-Project-Suite`. The suite is not importable yet when this
module loads. So we VENDOR the pure helpers here and reach the suite only
at node-EXECUTION time, via `_ensure_suite_patches()`, by which point every
pack has finished importing.

    H3 Loop Close  pin a clip's own opening frames at its tail so the body
                   must denoise back into the opening motion; report the
                   overlap as trim_frames.
    H3 Loop Trim   drop that trailing overlap (edge='tail') off the decoded
                   clip, picture and sound together, so nothing is doubled.
"""

import logging
import sys
import types

import node_helpers
import torch

_LOG = logging.getLogger("h3_loop")

try:
    # Only importable inside a running ComfyUI process (comfy/ is not an
    # installed package -- ComfyUI puts it on sys.path itself at startup).
    # tests/smoke.py loads this module standalone, so degrade gracefully:
    # INPUT_TYPES still gets a valid combo list containing our defaults.
    import comfy.samplers
    _SAMPLERS = comfy.samplers.KSampler.SAMPLERS
    _SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS
except (ImportError, ModuleNotFoundError, AttributeError) as exc:
    _LOG.warning(
        "h3_chainloop: comfy.samplers.KSampler.SAMPLERS/SCHEDULERS not "
        "importable (%s); falling back to a single default entry each "
        "('er_sde'/'beta'). If this happens inside a running ComfyUI, "
        "comfy.samplers likely changed shape and needs attention.", exc)
    _SAMPLERS = ["er_sde"]
    _SCHEDULERS = ["beta"]


# --------------------------------------------------------------------------
# Vendored pure helpers -- copied VERBATIM from ComfyUI-H3-Project-Suite
# nodes.py so this module has zero import-time dependency on that pack.
# Keep them byte-identical to the upstream so behavior stays bit-exact.
# --------------------------------------------------------------------------

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)

# Run lengths a single clip can offer as an overlap window; these land on the
# video VAE's downscale grid, so the pinned run ends exactly at a real frame.
CONTEXT_LENGTHS = [1, 5, 22, 39, 56]

# must match ComfyUI-H3-Project-Suite patch_layout.MC_KEY
MC_KEY = "motion_context_index"


def _pixel_frames(latent_t):
    """Pixel frames covered by latent_t latent steps."""
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(latent_t))


def _step_offsets(latent_t):
    """Pixel-frame index at which each latent step begins."""
    out, acc = [], 0
    for k in range(latent_t):
        out.append(acc)
        acc += FRAME_PER_TOKEN[k % 5]
    return out


def _streams_from_latent(latent):
    """Unpack an H3 AV latent into its contained streams.

    NestedTensor.__getitem__ broadcasts the index into every contained
    tensor rather than selecting one, so samples[0] would strip the batch
    dimension off both streams. unbind() returns the pair.
    """
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError(
            "h3_suite: expected a MiniMax H3 AV latent (a nested "
            "video/audio pair), got %r" % type(samples))
    if not parts:
        raise ValueError("h3_suite: AV latent contains no streams")
    return parts


def _video_from_latent(latent):
    """Pull the video stream out of an H3 AV latent."""
    video = _streams_from_latent(latent)[0]
    if video.ndim == 4:  # unbatched [C,T,H,W]
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError("h3_suite: expected video latent [B,C,T,H,W], "
                         "got shape %s" % (tuple(video.shape),))
    return video


# --------------------------------------------------------------------------
# Runtime dependency on the suite: the layout patch that lets interior
# keyframe anchors build. The suite activates it inline on first node run;
# we find its module (loaded after us) and trigger the same activation.
# --------------------------------------------------------------------------

def _ensure_suite_patches():
    """Activate ComfyUI-H3-Project-Suite's marker-gated H3 keyframe patch.

    Scans loaded modules for the suite's own module (its __file__ sits under
    the pack dir) exposing `_activate_inline_patches`, and calls it. This
    stands in for the suite's own `_activate_inline_patches()` call at the
    top of its node bodies -- our tail keyframes are interior anchors and
    ComfyUI rejects them unless that patch is live.
    """
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        path = getattr(mod, "__file__", None)
        if (path and "ComfyUI-H3-Project-Suite" in path
                and hasattr(mod, "_activate_inline_patches")):
            mod._activate_inline_patches()
            return
    raise RuntimeError(
        "ComfyUI-H3-Loop requires ComfyUI-H3-Project-Suite installed and "
        "loaded (it provides the arbitrary-index H3 keyframe patch).")


def _parse_prompts(prompts, num_clips):
    r"""Parse a prompt string into a list of prompts for each clip.

    Supports two formats:
    - Single shared prompt: returned replicated num_clips times
    - Per-clip prompts separated by "===" or "^\s*===\s*$": must match num_clips
    """
    parts = [p.strip() for p in prompts.split("\n===\n")]
    # also tolerate a line that is exactly === with surrounding spaces
    if len(parts) == 1:
        import re
        parts = [p.strip() for p in re.split(r"(?m)^\s*===\s*$", prompts) if p.strip()]
    parts = [p for p in parts if p]
    if len(parts) == 1:
        return parts * num_clips
    if len(parts) == num_clips:
        return parts
    raise ValueError(
        "h3_chainloop: got %d prompt sections for %d clips; provide either "
        "1 (shared) or exactly %d (one per clip)" % (len(parts), num_clips, num_clips))


class H3LoopClose:
    """Close an H3 clip into a perfect loop.

    Slice the FIRST `overlap_length` frames of a previously generated clip
    (`context_latent`) as never-denoised keyframe rows, pin that same slice
    at the TAIL of this generation so the body must denoise back into the
    opening motion, and (seed_head) hold the slice at the HEAD of the clip
    latent so departure and arrival share the same velocity. Report
    `trim_frames` = the overlap so H3 Loop Trim (edge='tail') can drop the
    replica; the kept clip then loops seamlessly with nothing doubled.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "latent": ("LATENT",),
                "context_latent": ("LATENT", {
                    "tooltip": "Pass-1 clip's SAMPLER OUTPUT AV latent. Its "
                               "first frames are sliced straight from the "
                               "latent (no VAE round trip) and become the "
                               "loop's shared start/end. Same resolution as "
                               "this clip."}),
                "overlap_length": (CONTEXT_LENGTHS, {
                    "default": 22,
                    "tooltip": "Frames of the clip's OWN start pinned at the "
                               "end. Bigger = smoother, more of the loop spent "
                               "re-treading the opening. Snaps to the VAE grid."}),
                "seed_head": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Write the start slice into the head of the "
                               "clip latent and hold it while sampling, so the "
                               "loop departs from the exact motion it returns "
                               "to. Wire this node's latent OUTPUT to the sampler."}),
                "head_hold": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How firmly the seeded head is held. 1.0 keeps "
                               "it exactly; lower lets the model repaint it "
                               "slightly, which can ease the release."}),
                "pin_head_keyframes": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Also emit the start slice as cond rows at the "
                               "head (belt-and-braces velocity anchor). Usually "
                               "seed_head is enough; enable if the seam drifts."}),
            },
            "optional": {
                "enabled": ("BOOLEAN", {
                    "forceInput": True,
                    "tooltip": "False passes conditioning through untouched "
                               "and reports trim_frames=0, so the whole loop "
                               "chain disarms off one boolean."}),
                "merge_context": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Merge these tail loop pins with any keyframes "
                               "already on the incoming conditioning -- e.g. a "
                               "head continue-pin from an upstream H3Context -- "
                               "so ONE clip can carry both a head pin and its "
                               "own tail loop, which is what lets a whole CHAIN "
                               "of clips loop back to clip 1. False = old "
                               "behaviour: replace whatever keyframes were "
                               "already set."}),
                "seed_tail": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "HARD-hold clip 1's start slice in this clip's "
                               "TAIL latent steps during sampling (temporal "
                               "inpaint), the mirror of seed_head at the other "
                               "end. This is what makes a chain-loop of N>=3 "
                               "clips close pixel-tight at the wrap instead of "
                               "drifting. Merges with any head-hold mask already "
                               "on the incoming latent (e.g. from H3Context), so "
                               "the head stays held too. Skipped (falls back to "
                               "cond-rows only) on off-grid clip lengths where "
                               "the tail phase would not line up."}),
                "tail_hold": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How firmly the seeded tail is held. 1.0 keeps "
                               "clip 1's start exactly at the wrap; lower lets "
                               "the model repaint it slightly."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "INT", "LATENT")
    RETURN_NAMES = ("conditioning", "trim_frames", "latent")
    FUNCTION = "apply"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = ("Pin a clip's own opening frames at its tail to make a "
                   "seamless loop; trim the overlap with H3 Loop Trim "
                   "(edge='tail').")

    def _head_blocks_from_latent(self, context_latent, overlap_length,
                                 target_video, frame_count):
        # phase-0 head slice: step 0 is index 0 (divisible by 5), so the first
        # `steps` steps cover _pixel_frames(steps) frames with the same
        # (1,4,4,4,4) coverage a fresh encode would produce.
        if context_latent is None:
            raise ValueError(
                "h3_loop: context_latent is not wired. Wire the pass-1 "
                "clip's sampler-output AV latent.")
        parts = _streams_from_latent(context_latent)
        ctx = parts[0]
        if getattr(ctx, "ndim", 0) == 4:
            ctx = ctx.unsqueeze(0)
        if getattr(ctx, "ndim", 0) != 5:
            raise ValueError(
                "h3_loop: context_latent video stream has shape %s, "
                "expected [B,C,T,H,W]." % (tuple(getattr(ctx, "shape", ())),))
        if (int(ctx.shape[3]) != int(target_video.shape[3])
                or int(ctx.shape[4]) != int(target_video.shape[4])):
            raise ValueError(
                "h3_loop: context_latent is %dx%d latent but this clip is "
                "%dx%d; the loop slice cannot resize. Render both at the same "
                "resolution." % (int(ctx.shape[4]), int(ctx.shape[3]),
                                  int(target_video.shape[4]),
                                  int(target_video.shape[3])))
        total = int(ctx.shape[2])
        want = int(overlap_length)
        # The head slice starts at phase-0 step 0, so any step count is a
        # valid phase-0 run; snap an off-grid request DOWN to the largest
        # offered overlap window (CONTEXT_LENGTHS) that fits, matching the
        # combo's values and the pack's snap-down convention.
        usable = [c for c in CONTEXT_LENGTHS if 0 < c <= want]
        if not usable:
            raise ValueError(
                "h3_loop: overlap_length %d is below the smallest usable "
                "window (%d)." % (want, min(CONTEXT_LENGTHS)))
        covered = max(usable)
        steps = 1
        while steps < total and _pixel_frames(steps) < covered:
            steps += 1
        if _pixel_frames(steps) != covered:
            raise ValueError(
                "h3_loop: context_latent is only %d latent steps, too "
                "short for a %d frame overlap." % (total, covered))
        if covered != want:
            _LOG.warning("h3_loop: overlap %d off grid; using %d "
                         "(first %d latent steps).", want, covered, steps)
        if covered >= frame_count:
            raise ValueError(
                "h3_loop: overlap %d must be a small fraction of the %d "
                "frame clip." % (covered, frame_count))
        blocks = [ctx[:, :, j:j + 1] for j in range(steps)]
        offsets = _step_offsets(steps)
        # let seed_head reuse the inpaint helper: head slice starts at step 0
        self._last_pin = (ctx, 0, steps)
        return blocks, offsets, covered

    def _seed_head_latent(self, latent, head_hold):
        """Write the pinned steps into the clip latent's head and attach a
        noise mask that holds them during sampling.

        ComfyUI's sampler re-noises masked regions to the current sigma each
        step (temporal inpainting), so the held head rides the schedule as
        the previous clip's content rather than being merely conditioned
        toward it. A non-nested mask applies to the VIDEO part; the sampler
        pads the audio part with ones, so audio keeps denoising freely.
        """
        # NB: unlike the suite we do NOT install its vendored PR-15375 mask
        # layer / _ensure_mask_layer boost here -- this runs STOCK seed_head.
        # The held head still lands bit-exact; it just conditions slightly
        # more weakly, so we avoid depending on the suite's mask_compat guts.
        import torch
        ctx, k0, steps = getattr(self, "_last_pin", (None, 0, 0))
        if ctx is None or steps <= 0:
            return latent
        samples = latent["samples"]
        parts = list(samples.unbind())
        video = parts[0].clone()
        # the pinned tail becomes the clip's opening, step for step
        video[:, :, 0:steps] = ctx[:, :, k0:k0 + steps]
        parts[0] = video
        mask = torch.ones(
            (int(video.shape[0]), 1, int(video.shape[2]),
             int(video.shape[3]), int(video.shape[4])),
            dtype=torch.float32)
        # mask semantics: 1 = denoise freely, 0 = hold to the latent
        mask[:, :, 0:steps] = 1.0 - head_hold
        out = dict(latent)
        out["samples"] = type(samples)(parts)
        out["noise_mask"] = mask
        _LOG.info("h3_loop: seeded %d head steps from the previous tail, "
                  "hold %.2f; wire this node's latent output into the "
                  "sampler", steps, head_hold)
        return out

    def _seed_tail_latent(self, latent, tail_hold):
        """Write the pinned start steps into the clip latent's TAIL and hold
        them during sampling -- the mirror of _seed_head_latent.

        seed_head hard-holds content at the HEAD so the loop departs from the
        opening; this hard-holds clip 1's START in this clip's TAIL so the loop
        ARRIVES exactly back on clip 1's opening. For an N>=3 chain-loop the
        soft cond-rows alone let the wrap drift; the hard tail hold closes it
        pixel-tight, exactly as seed_head closes the mid-chain junctions.

        PHASE SAFETY: H3 pixel coverage cycles by absolute step index mod 5
        (FRAME_PER_TOKEN). The captured slice is clip 1's phase-0 head steps
        [0:steps]. For it to line up when copied into the tail, the tail's
        first absolute step index (T-steps) must itself be a multiple of 5, so
        the tail steps carry the SAME (1,4,4,4,4...) coverage phase. For the
        grid lengths (clip T%5==2, overlap steps%5==2) T-steps is a multiple of
        5. If it is not (an off-grid clip length), copying would smear frames
        across the phase boundary, so we SKIP the tail write, warn, and leave
        the latent to the cond-rows-only path -- never corrupt it.

        NOISE MASK: mask semantics are 1=denoise freely, 0=hold. We MERGE with
        any noise_mask already on the incoming latent (H3Context seeds the HEAD
        and attaches a head-hold mask; seed_head here may add one too), cloning
        it and setting only the tail region so the head-held region survives.
        """
        import torch
        ctx, k0, steps = getattr(self, "_last_pin", (None, 0, 0))
        if ctx is None or steps <= 0:
            return latent
        samples = latent["samples"]
        parts = list(samples.unbind())
        video = parts[0].clone()
        latent_t = int(video.shape[2])
        # PHASE SAFETY: the tail window must begin on a step index divisible by
        # 5 so its coverage phase matches clip 1's phase-0 head slice.
        if (latent_t - steps) % 5 != 0:
            _LOG.warning(
                "h3_loop: seed_tail skipped -- clip is %d latent steps and the "
                "overlap is %d steps, so the tail window would begin at step %d "
                "(not a multiple of 5) and the copied start frames would not "
                "line up in phase. Falling back to cond-rows-only for the tail; "
                "the latent is left uncorrupted. Use a grid clip length.",
                latent_t, steps, latent_t - steps)
            return latent
        # the pinned start slice becomes the clip's closing, step for step
        video[:, :, latent_t - steps:latent_t] = ctx[:, :, k0:k0 + steps]
        parts[0] = video
        # MERGE with any existing noise_mask (head-hold from H3Context/seed_head)
        existing = latent.get("noise_mask")
        if existing is not None:
            mask = existing.clone()
        else:
            mask = torch.ones(
                (int(video.shape[0]), 1, int(video.shape[2]),
                 int(video.shape[3]), int(video.shape[4])),
                dtype=torch.float32)
        # mask semantics: 1 = denoise freely, 0 = hold to the latent. Only the
        # tail region is touched, so any head-held region is preserved.
        mask[:, :, latent_t - steps:latent_t] = 1.0 - tail_hold
        out = dict(latent)
        out["samples"] = type(samples)(parts)
        out["noise_mask"] = mask
        _LOG.info("h3_loop: seeded %d TAIL steps [%d:%d] from clip 1's start, "
                  "hold %.2f (merged with existing mask=%s); wire this node's "
                  "latent output into the sampler", steps, latent_t - steps,
                  latent_t, tail_hold, existing is not None)
        return out

    def apply(self, conditioning, latent, context_latent, overlap_length,
              seed_head=True, head_hold=1.0, pin_head_keyframes=False,
              enabled=True, merge_context=True, seed_tail=False,
              tail_hold=1.0):
        if enabled is False:
            _LOG.info("h3_loop: loop close disabled (chain inactive); "
                      "passing conditioning through untouched")
            return (conditioning, 0, latent)
        _ensure_suite_patches()

        video = _video_from_latent(latent)
        frame_count = _pixel_frames(int(video.shape[2]))
        blocks, offsets, covered = self._head_blocks_from_latent(
            context_latent, overlap_length, video, frame_count)

        keyframes = []
        # tail anchor: the start slice sits at the END of the timeline
        for o, blk in zip(offsets, blocks):
            keyframes.append({"resolved_frame_index": 0,
                              MC_KEY: (frame_count - covered) + o,
                              "latent": blk})
        if pin_head_keyframes:
            # optional head cond rows; skip index 0 (the base image cond,
            # if any, already pins pixel 0)
            for o, blk in zip(offsets, blocks):
                if o == 0:
                    continue
                keyframes.append({"resolved_frame_index": 0, MC_KEY: o,
                                  "latent": blk})

        # MERGE with any keyframes an upstream node already put on this
        # conditioning. node_helpers.conditioning_set_values REPLACES the
        # minimax_keyframes key, so wiring H3Context (a head continue-pin) ->
        # H3LoopClose (a tail loop-pin) on the same conditioning would clobber
        # the head pin. Reading it back and unioning is what lets ONE clip
        # carry both -- so the FINAL clip of a chain can close the loop to
        # clip 1 while still continuing from the previous clip.
        #
        # No pre-existing keyframes (standalone single-clip loop) -> pass the
        # list exactly as built above, byte-for-byte as before. minimax_refs
        # is never passed, so conditioning_set_values leaves any audio refs
        # H3Context set untouched (the keyframes+refs coexistence path).
        existing = []
        if merge_context and conditioning:
            existing = conditioning[0][1].get("minimax_keyframes") or []
        if existing:
            merged = list(existing) + keyframes
            # The suite's layout patch builds one cond span per keyframe in
            # LIST ORDER and assigns each span's timeline position from that
            # keyframe's MC_KEY (patch_layout._fixup), so keep the union sorted
            # by MC_KEY ascending -- head indices [0..k] land before tail
            # indices [frame_count-k..frame_count].
            merged.sort(key=lambda kf: kf.get(MC_KEY, 0))
            # Dedupe defensively on MC_KEY, keeping the LAST (the loop pin);
            # in normal use head and tail windows do not collide.
            deduped = {}
            for kf in merged:
                idx = kf.get(MC_KEY)
                if idx in deduped:
                    _LOG.warning(
                        "h3_loop: two keyframes share %s=%s; keeping the loop "
                        "pin (the later one). Head [0..k] and tail "
                        "[frame_count-k..] windows should not normally collide.",
                        MC_KEY, idx)
                deduped[idx] = kf
            merged = list(deduped.values())
            _LOG.info("h3_loop: merged %d upstream keyframe(s) with %d loop "
                      "pin(s) -> %d total (chain-loop path).",
                      len(existing), len(keyframes), len(merged))
        else:
            merged = keyframes

        out = node_helpers.conditioning_set_values(
            conditioning, {"minimax_keyframes": merged,
                           "minimax_frame_count": frame_count})

        out_latent = latent
        if seed_head:
            out_latent = self._seed_head_latent(latent, float(head_hold))
        if seed_tail:
            # Reads out_latent so, when seed_head also ran, seed_tail sees (and
            # merges with) the head-hold mask seed_head just wrote; in the chain
            # this reads H3Context's head-hold mask off the incoming latent.
            out_latent = self._seed_tail_latent(out_latent, float(tail_hold))
        _LOG.info("h3_loop: loop close, %d frame clip, overlap %d (%d steps), "
                  "%d keyframe rows, trim %d off the tail.",
                  frame_count, covered, len(blocks), len(merged), covered)
        return (out, covered, out_latent)


class H3LoopTrim:
    """Drop the trailing loop overlap off a decoded clip, picture and sound
    together.

    H3 Loop Close pins the clip's own opening frames at its TAIL, so those
    frames come back doubled at the end of the delivered clip. Trimming only
    the images would leave the audio a full trim_frames longer than the
    video, and muxing those puts the whole soundtrack out of sync by
    trim_frames/24 seconds. So this takes both streams and removes the same
    span from each: whole frames from the images, the matching number of
    samples from the waveform.

    edge='tail' (the loop default) drops the trailing overlap. edge='head'
    is the chaining behaviour -- drop the leading pinned frames instead.

    match_tail additionally truncates or zero-pads the audio so its
    duration equals frames/fps exactly: H3's 40 Hz audio grid rounds to the
    nearest step against 24 fps picture, shipping ~8 ms of excess or
    shortage on some lengths, which would otherwise accumulate down a chain.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "trim_frames": ("INT", {"default": 0, "min": 0, "max": 4096}),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Decoded audio for the same clip. Trimmed by the "
                               "matching duration so sound stays locked to "
                               "picture. Leave unwired for silent clips."}),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                    "tooltip": "Frame rate used to convert the trim into an "
                               "audio duration. Must match what you feed "
                               "Create Video."}),
                "match_tail": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Truncate or zero-pad audio so its duration "
                               "equals frames/fps exactly. H3 rounds its 40 Hz "
                               "audio grid to the nearest step, producing about "
                               "8ms of excess or shortage on some lengths."}),
                "edge": (["tail", "head"], {
                    "default": "tail",
                    "tooltip": "tail: drop the trailing loop overlap (H3 Loop "
                               "Close). head: drop leading pinned frames "
                               "(chaining)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "trim"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = ("Remove the trailing loop overlap from a decoded H3 clip, "
                   "trimming picture and sound by the same duration.")

    def trim(self, images, trim_frames, audio=None, fps=24.0, match_tail=True,
             edge="tail"):
        n = max(0, int(trim_frames))
        total = int(images.shape[0])
        if n >= total:
            raise ValueError(
                "h3_loop: asked to trim %d frames from a %d frame clip"
                % (n, total))
        if edge == "tail":
            out_images = images[:total - n] if n else images
        else:
            out_images = images[n:] if n else images

        out_audio = audio
        if audio is not None:
            waveform = audio["waveform"]
            sr = int(audio["sample_rate"])
            seconds = n / float(fps)
            cut = int(round(seconds * sr))
            length = int(waveform.shape[-1])
            if cut >= length:
                raise ValueError(
                    "h3_loop: trimming %.3fs from %.3fs of audio would "
                    "leave nothing. Check that fps matches the clip."
                    % (seconds, length / sr))
            waveform = waveform[..., :length - cut] if edge == "tail" \
                else waveform[..., cut:]

            if match_tail:
                frames_left = total - n
                want = int(round(frames_left / float(fps) * sr))
                have = int(waveform.shape[-1])
                if have > want:
                    over = have - want
                    waveform = waveform[..., :want]
                    _LOG.info("h3_loop: tail trimmed %d samples "
                              "(%.2fms) so audio matches %d frames exactly",
                              over, over / sr * 1000.0, frames_left)
                elif have < want:
                    missing = want - have
                    waveform = torch.nn.functional.pad(waveform, (0, missing))
                    _LOG.info("h3_loop: tail padded %d zero samples "
                              "(%.2fms) so audio matches %d frames exactly",
                              missing, missing / sr * 1000.0, frames_left)

            out_audio = {"waveform": waveform, "sample_rate": sr}
            _LOG.info("h3_loop: %d frames / %.4fs picture, %.4fs sound, "
                      "drift %.2fms",
                      total - n, (total - n) / float(fps),
                      int(waveform.shape[-1]) / sr,
                      abs((total - n) / float(fps) - int(waveform.shape[-1]) / sr) * 1000.0)
        elif n:
            _LOG.info("h3_loop: trimmed %d %s frames, %d remain. "
                      "No audio wired; if this clip has sound, mux it through "
                      "this node or it will drift from the picture.",
                      n, edge, total - n)

        return (out_images, out_audio)


def _clamp_length(n):
    """Clamp a requested clip length to the Ref2VA widget bounds.

    NOT a grid-snap: MiniMaxH3ReferenceToVideo snaps `length` UP internally
    via `align_frame_count` (next 17k+5), so this node must not pre-snap to
    the VIDEO_RUN_GRID -- only clamp to what the widget will accept.
    """
    return max(5, min(3600, int(n)))


def _cls(name):
    """Resolve a node CLASS by its NODE_CLASS_MAPPINGS key, at call time
    (not import time), so packages that load after this one are visible.

    Checks THIS module's own NODE_CLASS_MAPPINGS first -- this is a
    no-op detour for our own classes (H3LoopClose etc.) in production, but
    it is also the hook tests use to inject fakes for classes that live in
    OTHER packs (MiniMaxH3ReferenceToVideo, H3Context): monkeypatch
    `h3loop_nodes.NODE_CLASS_MAPPINGS[name] = FakeClass` and this function
    picks it up with no running ComfyUI needed. Falls back to the real,
    ComfyUI-aggregated registry (core `nodes.NODE_CLASS_MAPPINGS`, which by
    node-execution time has every installed pack's classes merged into it)
    for anything not found locally -- this is where
    MiniMaxH3ReferenceToVideo/H3Context actually resolve in a real run.
    """
    c = NODE_CLASS_MAPPINGS.get(name)
    if c is None:
        try:
            import nodes as _n
        except ImportError:
            _n = None
        if _n is not None:
            c = _n.NODE_CLASS_MAPPINGS.get(name)
    if c is None:
        raise RuntimeError(
            "h3_chainloop: required node %s not found — is its pack "
            "installed?" % name)
    return c


# MiniMaxH3ReferenceToVideo's `ref_images` Autogrow bundle names its slots
# f"{prefix}{i}" for i in range(max) (comfy_api/latest/_io.py,
# Autogrow.TemplatePrefix.names). The node declares that template as
# `io.Autogrow.Input("ref_images", template=io.Autogrow.TemplatePrefix(
# input=io.Image.Input("ref_image"), prefix="ref_image_", min=0, max=9))`
# (comfy_extras/nodes_minimax_h3.py lines 267-270) -- prefix already ends in
# "_", so slot 0 is "ref_image_0" (NOT "ref_image0"). Confirmed by reading
# both source files directly (not inferred from the ambiguous name alone);
# execute() only ever iterates `.values()` (line 295) so the key content
# does not affect behavior today, but the wire-accurate name is used here in
# case a future version keys off it (e.g. per-slot ordering/labels).
# The model accepts up to 9 identity references (slots ref_image_0..8);
# H3ChainLoop exposes all 9 and feeds every wired one to every clip.
_REF_PREFIX = "ref_image_"
_MAX_REFS = 9  # slots ref_image_0 .. ref_image_8


def _ref_bundle(ref_images):
    """Build MiniMaxH3ReferenceToVideo's `ref_images` Autogrow bundle from an
    ordered list of IMAGE inputs. Unwired (None) slots are skipped; each
    image keeps its slot index as its key (ref_image_0..ref_image_8), so a
    gap in the middle is preserved rather than compacted. Insertion order
    follows slot order, which is also the order execute() sees via .values()."""
    return {"%s%d" % (_REF_PREFIX, i): img
            for i, img in enumerate(ref_images) if img is not None}


def _sample(model, cond, latent, sampler_name, scheduler, steps, denoise, seed):
    """Sample one clip's latent.

    Inline CFGGuider form (docs/API_NOTES.md §2): reproduces
    SamplerCustomAdvanced.execute's internals directly instead of importing
    comfy_extras.nodes_custom_sampler (its RandomNoise/KSamplerSelect/
    BasicScheduler/SamplerCustomAdvanced are V3 io.ComfyNode classmethod
    nodes, not meant to be instantiated/called from other Python). Broken
    out as its own module-level function so tests can monkeypatch
    `h3loop_nodes._sample` wholesale -- no GPU or loaded model needed to
    unit-test the orchestration that calls it.
    """
    import comfy.model_management
    import comfy.sample
    import comfy.samplers
    import comfy.utils
    import latent_preview

    guider = comfy.samplers.CFGGuider(model)
    guider.set_cfg(1.0)
    guider.inner_set_conds({"positive": cond})

    sampler = comfy.samplers.sampler_object(sampler_name)

    total_steps = steps if denoise >= 1.0 else int(steps / denoise)
    sigmas = comfy.samplers.calculate_sigmas(
        model.get_model_object("model_sampling"), scheduler, total_steps
    ).cpu()
    sigmas = sigmas[-(steps + 1):]

    latent = latent.copy()
    latent_image = comfy.sample.fix_empty_latent_channels(
        guider.model_patcher, latent["samples"],
        latent.get("downscale_ratio_spacial"),
        latent.get("downscale_ratio_temporal"))
    latent["samples"] = latent_image
    noise_mask = latent.get("noise_mask")
    noise = comfy.sample.prepare_noise(latent_image, seed, latent.get("batch_index"))

    x0_output = {}
    callback = latent_preview.prepare_callback(
        guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    samples = guider.sample(
        noise, latent_image, sampler, sigmas, denoise_mask=noise_mask,
        callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples = samples.to(comfy.model_management.intermediate_device())

    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    return out


class H3ChainLoop:
    """Orchestrate an N-clip MiniMax H3 Ref2VA seamless loop.

    num_clips=1 renders a two-pass single loop; num_clips>=2 chains clips
    (H3Context continue-pin between clips, H3LoopClose+seed_tail closing the
    last clip back to clip 1), then decodes, trims the overlaps, and
    concatenates. Output frames ~= num_clips*(length-overlap).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "ref_image_0": ("IMAGE", {
                    "tooltip": "Primary identity reference (required). "
                               "Ref2VA uses it for identity only, not the "
                               "scene. Add ref_image_1..8 for extra views."}),
                "prompts": ("STRING", {
                    "multiline": True,
                    "default": "a scene continues seamlessly",
                    "tooltip": "One shared prompt for every clip, or "
                               "per-clip prompts separated by a line "
                               "containing only ===; the count must then "
                               "match num_clips exactly."}),
                "num_clips": ("INT", {"default": 2, "min": 1, "max": 12}),
                "length": ("INT", {
                    "default": 124, "min": 5, "max": 3600,
                    "tooltip": "Frames per clip. MiniMaxH3ReferenceToVideo "
                               "snaps this UP internally to the next valid "
                               "grid length; this node only clamps it to "
                               "the widget bounds."}),
                "overlap": (CONTEXT_LENGTHS, {"default": 22}),
                "width": ("INT", {"default": 768, "min": 32, "step": 32}),
                "height": ("INT", {"default": 1344, "min": 32, "step": 32}),
                "steps": ("INT", {"default": 8, "min": 1}),
                "sampler_name": (_SAMPLERS, {"default": "er_sde"}),
                "scheduler": (_SCHEDULERS, {"default": "beta"}),
                "denoise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "seed_tail": ("BOOLEAN", {"default": True}),
                "head_hold": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0}),
                "tail_hold": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "%s%d" % (_REF_PREFIX, i): ("IMAGE", {
                    "tooltip": "Extra identity reference (optional). Every "
                               "wired reference feeds every clip in the "
                               "chain (identity is constant across the loop)."})
                for i in range(1, _MAX_REFS)
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "sampling/minimax"
    DESCRIPTION = ("Chain N MiniMax H3 Ref2VA clips into one seamless loop, "
                   "orchestrating H3 Loop Close/Trim across the chain.")

    def run(self, model, clip, vae, ref_image_0, prompts, num_clips, length,
            overlap, width, height, steps, sampler_name, scheduler, denoise,
            seed, seed_tail, head_hold, tail_hold,
            ref_image_1=None, ref_image_2=None, ref_image_3=None,
            ref_image_4=None, ref_image_5=None, ref_image_6=None,
            ref_image_7=None, ref_image_8=None):
        if width % 32 or height % 32:
            raise ValueError(
                "h3_chainloop: width and height must each be divisible by 32 "
                "(got %dx%d); 9:16=768x1344, 16:9=1344x768" % (width, height))
        if overlap not in CONTEXT_LENGTHS:
            raise ValueError(
                "h3_chainloop: overlap must be one of %s" % (CONTEXT_LENGTHS,))
        if overlap >= length:
            raise ValueError(
                "h3_chainloop: overlap (%d) must be < length (%d)"
                % (overlap, length))
        if num_clips >= 2 and 2 * overlap >= length:
            raise ValueError(
                "h3_chainloop: for num_clips>=2, 2*overlap (%d) must be < length "
                "(%d) — the final clip is trimmed on both ends" % (2 * overlap, length))
        length = _clamp_length(length)

        sections = _parse_prompts(prompts, num_clips)
        ref_images = [ref_image_0, ref_image_1, ref_image_2, ref_image_3,
                      ref_image_4, ref_image_5, ref_image_6, ref_image_7,
                      ref_image_8]
        ctx = types.SimpleNamespace(
            model=model, clip=clip, vae=vae, ref_images=ref_images,
            width=width, height=height, length=length, overlap=overlap,
            steps=steps, sampler_name=sampler_name, scheduler=scheduler,
            denoise=denoise, seed_tail=seed_tail, head_hold=head_hold,
            tail_hold=tail_hold)

        if num_clips == 1:
            # Two-pass single loop: harvest a clip, then regenerate it
            # against its own harvest as the loop target so the body denoises
            # back into its own opening. Only the second (loop) pass is
            # decoded/returned; the harvest exists purely to seed the pin.
            harvest, _, _ = self._gen_clip(
                ctx, sections[0], context_latent=None, loop_target_latent=None,
                seed=seed)
            loop_lat, _, trim_tail = self._gen_clip(
                ctx, sections[0], context_latent=None,
                loop_target_latent=harvest, seed=seed + 1)
            images = _cls("VAEDecode")().decode(vae, loop_lat)[0]
            if trim_tail:
                images = _cls("H3LoopTrim")().trim(
                    images, trim_tail, edge="tail")[0]
            return (images,)

        # N>=2 chain: clip 1 seeds the chain and the eventual loop target;
        # each subsequent clip continues from the previous clip (head trim
        # drops the continue re-tread) and the LAST clip also closes the
        # loop back to clip 1 (tail trim drops the loop replica).
        prev, _, _ = self._gen_clip(
            ctx, sections[0], context_latent=None, loop_target_latent=None,
            seed=seed)
        loop_target = prev
        frames = [_cls("VAEDecode")().decode(vae, prev)[0]]

        for i in range(1, num_clips):
            is_last = (i == num_clips - 1)
            cur, trim_head, trim_tail = self._gen_clip(
                ctx, sections[i], context_latent=prev,
                loop_target_latent=(loop_target if is_last else None),
                seed=seed + i)
            img = _cls("VAEDecode")().decode(vae, cur)[0]
            if trim_head:
                img = _cls("H3LoopTrim")().trim(img, trim_head, edge="head")[0]
            if is_last and trim_tail:
                img = _cls("H3LoopTrim")().trim(img, trim_tail, edge="tail")[0]
            frames.append(img)
            prev = cur

        return (torch.cat(frames, dim=0),)

    def _gen_clip(self, ctx, prompt, context_latent, loop_target_latent, seed):
        """Build Ref2VA conditioning for one clip, optionally apply the
        continue-pin (H3Context) and/or loop-close (H3LoopClose), sample,
        and return the result with trim counts.

        `ctx` is any attribute-holder (e.g. types.SimpleNamespace) bundling
        this run's fixed-per-chain settings: model, clip, vae, ref_images
        (an ordered list of up to 9 IMAGE refs, None for empty slots),
        width, height, length, overlap, steps, sampler_name, scheduler,
        denoise, seed_tail, head_hold, tail_hold.

        `context_latent`: previous clip's sampler-output AV latent to
        continue from (wired into H3Context as a head pin), or None for the
        first clip in the chain.
        `loop_target_latent`: clip-1 sampler-output AV latent to pin at this
        clip's tail (wired into H3LoopClose), or None if this clip does not
        close the loop.

        Returns `(out_latent, trim_head, trim_tail)`: `trim_head` is
        H3Context's reported trim_frames (0 if context_latent was None,
        i.e. no continue-pin ran); `trim_tail` is H3LoopClose's reported
        trim_frames (0 if loop_target_latent was None).
        """
        ref2va = _cls("MiniMaxH3ReferenceToVideo")
        out = ref2va.execute(
            clip=ctx.clip, prompt=prompt, width=ctx.width, height=ctx.height,
            length=ctx.length, vae=ctx.vae,
            ref_images=_ref_bundle(ctx.ref_images))
        # io.NodeOutput supports __getitem__ but not tuple-unpack in every
        # comfy_api version; index explicitly (also works for a plain tuple,
        # which is what the mocked tests return).
        cond, latent = out[0], out[1]

        trim_head = 0
        if context_latent is not None:
            cond, trim_head, latent = _cls("H3Context")().apply(
                conditioning=cond, latent=latent, context_length=ctx.overlap,
                encode_mode="video", anchor_mode="head", crop="disabled",
                video_source="latent", context_latent=context_latent,
                seed_head=True, head_hold=ctx.head_hold)

        trim_tail = 0
        if loop_target_latent is not None:
            cond, trim_tail, latent = _cls("H3LoopClose")().apply(
                conditioning=cond, latent=latent,
                context_latent=loop_target_latent, overlap_length=ctx.overlap,
                seed_head=(context_latent is None), head_hold=ctx.head_hold,
                merge_context=True, seed_tail=ctx.seed_tail,
                tail_hold=ctx.tail_hold, pin_head_keyframes=False)

        out_latent = _sample(
            ctx.model, cond, latent, ctx.sampler_name, ctx.scheduler,
            ctx.steps, ctx.denoise, seed)
        return out_latent, trim_head, trim_tail


NODE_CLASS_MAPPINGS = {
    "H3LoopClose": H3LoopClose,
    "H3LoopTrim": H3LoopTrim,
    "H3ChainLoop": H3ChainLoop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3LoopClose": "H3 Loop Close",
    "H3LoopTrim": "H3 Loop Trim",
    "H3ChainLoop": "H3 Chain Loop",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
