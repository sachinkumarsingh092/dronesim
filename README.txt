# dronesim

Re-skin drone footage into a storybook cartoon. Objects (cars, trucks, people)
are detected, segmented, and re-painted with a diffusion model while the
background is left untouched. All GPU work runs on [Modal](https://modal.com).

**Stack:** GroundingDINO (open-vocab detection) → SAM2 (segmentation/tracking)
→ Stable Diffusion + ControlNet (canny-guided img2img re-style).

## Setup

```bash
uv sync                 
uv pip install modal
modal token new
```

## Pipeline

Each stage is a standalone Modal app. Run from the repo root; outputs land in `outputs/`.

| Stage | Script | Input → Output | What it does |
|-------|--------|----------------|--------------|
| Detect + segment | `pipeline/detect_image.py` | image → `det_out.png` | DINO boxes + SAM2 masks on one frame |
| Auto-segment | `pipeline/segment_image.py` | image → `sam_out.png` | SAM2 automatic mask generator (no prompt) |
| Track | `pipeline/track_video.py` | video → `track_out.mp4` | Seed boxes on frame 0, propagate masks through the clip |
| Cartoonize | `pipeline/cartoonize.py` | video → `cartoon_out.mp4` | ControlNet img2img re-style of the **whole** frame |
| Re-skin | `pipeline/reskin.py` | video → `reskin_out.mp4` | **Main:** re-style **only** detected objects, composite over real background |

## Run

```bash
# main deliverable
modal run pipeline/reskin.py --video-path data/clips/scene2.mp4

# single-frame sanity check
modal run pipeline/detect_image.py --image-path <frame.png>
```

Prompts are overridable, e.g. `--prompt "..."` and `--det-prompt "car . truck ."`.

