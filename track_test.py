import modal

app = modal.App("dronesim-track")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "torchvision")
    .env({"SAM2_BUILD_CUDA": "0"})
    .pip_install(
        "opencv-python-headless", "numpy", "huggingface_hub",
        "transformers", "pillow",
        "git+https://github.com/facebookresearch/sam2.git",
    )
)

vol = modal.Volume.from_name("hf-cache", create_if_missing=True)

@app.function(gpu="A100", image=image, timeout=1800, volumes={"/root/.cache/huggingface": vol})
def track(video_bytes: bytes, prompt: str) -> bytes:
    import os, numpy as np, cv2, torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from sam2.build_sam import build_sam2_video_predictor_hf

    # 1. mp4 -> jpg frames (SAM2 video wants a dir of <idx>.jpg)
    os.makedirs("/tmp/frames", exist_ok=True)
    open("/tmp/in.mp4", "wb").write(video_bytes)
    cap, frames, i = cv2.VideoCapture("/tmp/in.mp4"), [], 0
    while True:
        ok, f = cap.read()
        if not ok: break
        cv2.imwrite(f"/tmp/frames/{i}.jpg", f); frames.append(f); i += 1
    cap.release()
    h, w = frames[0].shape[:2]

    # 2. GroundingDINO on frame 0 -> seed boxes
    SCALE = 3
    big = cv2.resize(frames[0], None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    pil0 = Image.fromarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB))
    proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    gd = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to("cuda")
    inp = proc(images=pil0, text=prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = gd(**inp)
    res = proc.post_process_grounded_object_detection(
        out, inp.input_ids, threshold=0.15, text_threshold=0.15,
        target_sizes=[pil0.size[::-1]])[0]
    boxes = res["boxes"].cpu().numpy() / SCALE   # back to original coords


    # 3. SAM2 video: seed frame 0, propagate
    predictor = build_sam2_video_predictor_hf("facebook/sam2-hiera-tiny")
    state = predictor.init_state(video_path="/tmp/frames")
    for oid, box in enumerate(boxes):
        predictor.add_new_points_or_box(state, frame_idx=0, obj_id=oid, box=box)
    seg = {}
    for fidx, oids, logits in predictor.propagate_in_video(state):
        seg[fidx] = [(oid, (logits[k] > 0).cpu().numpy().squeeze())
                     for k, oid in enumerate(oids)]

    # 4. overlay tracked masks -> mp4
    rng = np.random.default_rng(0)
    colors = {oid: rng.integers(0, 255, 3) for oid in range(len(boxes))}
    vw = cv2.VideoWriter("/tmp/out.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 6, (w, h))
    for fidx in range(len(frames)):
        ov = frames[fidx].copy()
        for oid, m in seg.get(fidx, []):
            ov[m] = colors[oid]
        vw.write(cv2.addWeighted(frames[fidx], 0.5, ov, 0.5, 0))
    vw.release()
    return open("/tmp/out.mp4", "rb").read()

@app.local_entrypoint()
def main(video_path: str, prompt: str = "car . van . truck . bus . bicycle . person ."):
    data = open(video_path, "rb").read()
    open("track_out.mp4", "wb").write(track.remote(data, prompt))
    print("wrote track_out.mp4")

