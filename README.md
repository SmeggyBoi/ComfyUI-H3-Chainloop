# ComfyUI-H3-Chainloop

Seamless-loop nodes for **MiniMax H3** video. The headline node, **H3 Chain
Loop**, chains any number of Ref2VA clips into a single video that loops
*perfectly* — the last frame flows back into the first with nothing doubled at
the seam — with an optional different action per clip and up to 9 identity
reference images.

It works by pinning a clip's own opening frames as never-denoised keyframes at
its tail, so the model denoises the connecting motion *back into its own start*,
then trimming the overlap. For multi-clip chains it continues each clip from
the previous one and closes the last clip back to clip 1.

> **This is a sidecar to
> [ComfyUI-H3-Project-Suite](https://github.com/Adudeguyman/ComfyUI-H3-Project-Suite).**
> That pack is **required at runtime**: it installs the marker-gated H3 keyframe
> layout patch that lifts H3's first/last-only anchor restriction, and the
> loop's interior tail keyframes only build once that patch is live. Install
> both packs.

---

## What you get

| Node | Category | Role |
|------|----------|------|
| **H3 Chain Loop** (`H3ChainLoop`) | `sampling/minimax` | **Start here.** One node, integer `num_clips`, delimited per-clip prompts, up to 9 refs → a finished looping `IMAGE` batch. |
| **H3 Loop Close** (`H3LoopClose`) | `conditioning/minimax` | Building block: pins a clip's start slice at the tail (and optionally holds it at the head) so the body denoises back into the opening. |
| **H3 Loop Trim** (`H3LoopTrim`) | `conditioning/minimax` | Building block: drops the loop/continue overlap off the decoded frames (and matching audio). |

Most users only need **H3 Chain Loop** — it orchestrates the other two
internally. The building blocks are exposed for hand-built graphs.

---

## Requirements

1. **ComfyUI** with MiniMax H3 support (a recent build; this pack targets
   `>=0.33.1`).
2. **[ComfyUI-H3-Project-Suite](https://github.com/Adudeguyman/ComfyUI-H3-Project-Suite)**
   installed alongside this pack (runtime dependency, see above).
3. **MiniMax H3 model files** in your ComfyUI `models/` folders — a Ref2VA
   UNet, the H3 text encoder, the H3 video VAE, and (recommended) the H3
   Ref2VA acceleration LoRA. See the
   [MiniMax H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3) for
   the weights. The included example workflow references one working set of
   filenames; substitute whatever H3 weights you have.

No extra Python dependencies — everything else ships with ComfyUI.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Adudeguyman/ComfyUI-H3-Project-Suite
git clone https://github.com/SmeggyBoi/ComfyUI-H3-Chainloop
```

Restart ComfyUI. The three nodes appear under `sampling/minimax` and
`conditioning/minimax`. (Or install via ComfyUI-Manager once listed.)

---

## Quick start

1. Load `example_workflows/H3 Chain Loop.json`.
2. Point **Load Image** at your reference image (identity only — the scene
   comes from the prompt).
3. Set the model loaders to your H3 weights (UNet / text encoder / VAE / LoRA).
4. Set **num_clips**, the **duration** float (seconds), the **megapixel**
   float, and the **prompt** — then run.

The example workflow wires convenience inputs so you don't do math by hand:

- **Duration (seconds) → frames.** A `Float` (e.g. `8.0`) feeds a Math
  Expression `round(a * 24)` (H3 runs at 24 fps) into `length`. The node
  clamps `length` to 5–3600 and H3 snaps it up to the next valid grid length
  internally, so any seconds value is safe.
- **Megapixel → width/height.** A `Float` (e.g. `0.50`) feeds a Resolution
  Selector whose `multiple` is fixed at **32**, so width and height are always
  ÷32 (an H3 hard requirement — 16 VAE downscale × 2 DiT patch). Pick the
  aspect from the dropdown (9:16 → 768×1344, 16:9 → 1344×768, …).

### Prompts (one box, optional per-clip)

The `prompts` box takes **one prompt for every clip**, or **one action per
clip**:

- **1 section** → the same prompt drives every clip.
- **N sections**, separated by a line containing only `===`, where **N ==
  num_clips** → one action per clip, in order. Design the actions as a *cycle*
  so the last clip ends on clip 1's opening pose and the whole thing loops.

Any other section count is rejected with a clear error.

### Reference images (1–9)

`ref_image_0` is required; `ref_image_1 … ref_image_8` are optional (mirroring
`MiniMaxH3ReferenceToVideo`'s 9-slot bundle). Every wired reference feeds every
clip — identity stays constant across the loop (Ref2VA references are
identity-only, not per-clip scenes).

---

## `H3 Chain Loop` inputs

| Input | Meaning |
|-------|---------|
| `model`, `clip`, `vae` | H3 UNet chain / text encoder / video VAE. |
| `ref_image_0` (+ `ref_image_1..8`) | Identity reference(s), 1–9. |
| `prompts` | One shared prompt, or N sections split on a `===` line. |
| `num_clips` | 1 = single 2-pass loop; ≥2 = chain that loops as a whole. |
| `length` | Frames per clip (H3 snaps up to its grid; node clamps 5–3600). |
| `overlap` | Junction/loop overlap; one of `1, 5, 22, 39, 56`. Must be `< length` (and `2*overlap < length` for chains). |
| `width`, `height` | Must each be ÷32. |
| `steps`, `sampler_name`, `scheduler`, `denoise` | Sampling (defaults 8 / er_sde / beta / 1.0). |
| `seed`, `seed_tail`, `head_hold`, `tail_hold` | Seed + loop-closure controls. |

**Output:** a single `IMAGE` batch — wire it to a video combiner. Output length
≈ `num_clips * (length − overlap)` (for `num_clips=1`, `length − overlap`).

---

## How the loop closes (short version)

- **Single loop (`num_clips=1`)** is 2-pass: render a clip to harvest a moving
  start slice, then regenerate with that slice pinned at the tail (`H3LoopClose`,
  `seed_tail`) and trim the overlap. The last frame is the same moving pose as
  the first.
- **Chain (`num_clips≥2`)**: clip 1 is kept whole; each later clip continues
  from the previous clip's latent (head-pin, head-trimmed) and the final clip
  *also* closes back to clip 1 (tail-pin + `seed_tail`, tail-trimmed); the
  decoded segments are concatenated.

Tips for a tight seam: use an inherently **cyclical** action, keep lighting and
framing constant, and prefer a shorter per-clip horizon.

---

## Tests

Pure-Python, GPU-free (mocks the model/sampler):

```bash
python tests/smoke.py
```

---

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).

## Credits & inspiration

This project was **inspired by and builds directly on
[ComfyUI-H3-Project-Suite](https://github.com/Adudeguyman/ComfyUI-H3-Project-Suite)
by Adudeguyman.** The whole loop technique rests on the Project Suite's H3
context/keyframe work — in particular its arbitrary-index keyframe layout patch
(`patch_layout`), which lifts H3's first/last-only anchor restriction and lets
this pack place a clip's own start frames as *interior* keyframes at the tail.
The `H3 Loop Close` node grew out of studying the Suite's `H3Context`, and it
still calls into the Suite's patch at run time (which is why the Suite is a hard
runtime dependency). This project is packaged as a **sidecar** precisely so the
Project Suite can keep being updated upstream without colliding with the loop
work here. Huge thanks to that project — go star it.

Also built on [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3).
