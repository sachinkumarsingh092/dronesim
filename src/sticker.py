import modal

app = modal.App("dronesim-sticker")
vol = modal.Volume.from_name("hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch", "torchvision")
    .env({"SAM2_BUILD_CUDA": "0"})
    .pip_install(
        "diffusers", "transformers", "accelerate", "safetensors", "sentencepiece",
        "opencv-python-headless", "numpy", "pillow", "huggingface_hub",
        "ultralytics", "lapx", "simple-lama-inpainting", "einops",
        "git+https://github.com/facebookresearch/sam2.git",
    )
)

VEHICLES = ["car", "van", "truck", "bus"]

@app.function(gpu="A100", image=image, timeout=1200,
              secrets=[modal.Secret.from_name("huggingface")],
              volumes={"/root/.cache/huggingface": vol})
def make_assets(classes: list[str]) -> dict[str, bytes]:
    import io, torch
    from PIL import Image, ImageChops, ImageDraw, ImageFilter
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()
    SENT = (255, 0, 255)
    assets = {}
    for cls in classes:
        prompt = (f"flat 2D cartoon illustration of a {cls}, thick clean outlines, "
                  f"vibrant flat colors, drone view, centered, "
                  f"isolated on a plain solid white background, simple, sticker style")
        g = torch.Generator("cpu").manual_seed(0)
        img = pipe(prompt, guidance_scale=0.0, num_inference_steps=4,
                   max_sequence_length=256, width=1024, height=1024,
                   generator=g).images[0].convert("RGB")

        w, h = img.size
        work = img.copy()
        for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
            ImageDraw.floodfill(work, seed, SENT, thresh=90)
        alpha = (ImageChops.difference(work, Image.new("RGB", (w, h), SENT))
                 .convert("L").point(lambda v: 0 if v < 2 else 255)
                 .filter(ImageFilter.GaussianBlur(1)))  # soften the edge

        rgba = img.convert("RGBA"); rgba.putalpha(alpha)
        buf = io.BytesIO(); rgba.save(buf, format="PNG")
        assets[cls] = buf.getvalue()
    return assets


@app.function(gpu="A100", image=image, timeout=1800,
              volumes={"/root/.cache/huggingface": vol})
def apply_stickers(video_bytes: bytes, assets: dict[str, bytes]) -> bytes:
    import io, numpy as np, cv2, torch
    from collections import defaultdict
    from PIL import Image
    from ultralytics import YOLO
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    def tight_crop(rgba):  # trim transparent margin so the vehicle fills its box
        ys, xs = np.where(rgba[..., 3] > 8)
        return rgba if len(xs) == 0 else rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    stickers = {cls: tight_crop(np.array(Image.open(io.BytesIO(png)).convert("RGBA")))
                for cls, png in assets.items()}

    def paste(frame, rgba, box):
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, frame.shape[1]), min(y1, frame.shape[0])
        if x1 - x0 < 2 or y1 - y0 < 2:
            return
        s = cv2.resize(rgba, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
        s_bgr = cv2.cvtColor(s[..., :3], cv2.COLOR_RGB2BGR).astype(np.float32)
        a = s[..., 3:4].astype(np.float32) / 255.0
        roi = frame[y0:y1, x0:x1].astype(np.float32)
        frame[y0:y1, x0:x1] = (s_bgr * a + roi * (1 - a)).astype(np.uint8)

    open("/tmp/in.mp4", "wb").write(video_bytes)
    cap = cv2.VideoCapture("/tmp/in.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 12
    cap.release()

    # ── pass 1: YOLO + ByteTrack → per-frame [(id, class, box)] + the frames ──
    COCO = {2: "car", 5: "bus", 7: "truck"}  # COCO ids → our sticker classes
    model = YOLO("yolo11x.pt")
    results = model.track(source="/tmp/in.mp4", stream=True, persist=True,
                          classes=list(COCO), imgsz=1920, conf=0.25,
                          tracker="bytetrack.yaml", verbose=False)
    frames, dets = [], []
    for r in results:
        frames.append(r.orig_img.copy())
        fd = []
        if r.boxes is not None and r.boxes.id is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for b, i, c in zip(xyxy, ids, cls):
                fd.append((int(i), COCO[int(c)], b))
        dets.append(fd)
    h, w = frames[0].shape[:2]
    max_area = 0.12 * h * w  
    boxes_of = defaultdict(dict)                       
    votes = defaultdict(lambda: defaultdict(int))      
    for fi, fd in enumerate(dets):
        for i, key, b in fd:
            if (b[2] - b[0]) * (b[3] - b[1]) <= max_area:
                boxes_of[i][fi] = b
                votes[i][key] += 1
    key_of, size_of, centre_of = {}, {}, defaultdict(dict)
    for i, fb in boxes_of.items():
        key_of[i] = max(votes[i], key=votes[i].get)
        ws = sorted(b[2] - b[0] for b in fb.values())
        hs = sorted(b[3] - b[1] for b in fb.values())
        size_of[i] = (ws[len(ws) // 2], hs[len(hs) // 2])  
        fis = sorted(fb)
        cx = [(fb[f][0] + fb[f][2]) / 2 for f in fis]
        cy = [(fb[f][1] + fb[f][3]) / 2 for f in fis]
        for j, f in enumerate(fis):                        
            wx, wy = cx[max(0, j - 2):j + 3], cy[max(0, j - 2):j + 3]
            centre_of[i][f] = (sum(wx) / len(wx), sum(wy) / len(wy))

    def extent(i):
        xs = [c[0] for c in centre_of[i].values()]
        ys = [c[1] for c in centre_of[i].values()]
        return np.hypot(max(xs) - min(xs), max(ys) - min(ys))
    moving = {i for i in boxes_of if extent(i) >= max(30.0, 1.5 * max(size_of[i]))}

    from transformers import AutoModel
    tips = AutoModel.from_pretrained("google/tipsv2-b14", trust_remote_code=True).to("cuda").eval()
    ROAD = ["road", "street", "asphalt", "highway", "intersection", "crosswalk"]
    OTHER = ["building", "rooftop", "grass", "tree", "parking lot", "sidewalk", "field", "car"]
    with torch.no_grad():
        temb = torch.nn.functional.normalize(tips.encode_text(ROAD + OTHER).to("cuda"), dim=-1)
    n_road = len(ROAD)

    def road_grid(frame):  # 32x32 bool, True where a road class wins
        rgb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (448, 448))
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0).to("cuda")
        with torch.no_grad():
            pt = torch.nn.functional.normalize(tips.encode_image(t).patch_tokens[0], dim=-1)
            cls = (pt @ temb.T).argmax(-1).cpu().numpy().reshape(32, 32)
        return cv2.dilate((cls < n_road).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

    road_masks = [road_grid(f) for f in frames]

    def on_road(i):  
        hits = sum(int(road_masks[fi][min(int((b[1] + b[3]) / 2 / h * 32), 31),
                                       min(int((b[0] + b[2]) / 2 / w * 32), 31)])
                   for fi, b in boxes_of[i].items())
        return hits / len(boxes_of[i]) >= 0.5

    keep = {i for i in moving if on_road(i)}

    sample = frames[::max(1, len(frames) // 40)]
    plate = np.median(np.stack(sample), axis=0).astype(np.uint8)  # median over time = clean road
    sam = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")
    vw = cv2.VideoWriter("/tmp/out.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fi, frame in enumerate(frames):
        present = [i for i in keep if fi in boxes_of[i]]

        # erase the real cars: tight SAM2 silhouette → LaMa fill → feathered blend
        if present:
            sam.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            vmask = np.zeros((h, w), bool)
            for i in present:
                m, _, _ = sam.predict(box=np.asarray(boxes_of[i][fi])[None, :],
                                      multimask_output=False)
                vmask |= m[0].astype(bool)
            if vmask.any():
                m8 = cv2.dilate(vmask.astype(np.uint8) * 255, np.ones((7, 7), np.uint8))
                a = (cv2.GaussianBlur(m8, (9, 9), 0).astype(np.float32) / 255.0)[..., None]
                frame = (plate * a + frame * (1 - a)).astype(np.uint8)  # fill with clean road plate

        for i in sorted(present, key=lambda i: size_of[i][0] * size_of[i][1]):
            cx, cy = centre_of[i][fi]
            sw, sh = size_of[i]
            paste(frame, stickers[key_of[i]], (cx - sw / 2, cy - sh / 2, cx + sw / 2, cy + sh / 2))
        vw.write(frame)
    vw.release()
    return open("/tmp/out.mp4", "rb").read()


@app.local_entrypoint()
def main(video_path: str, regen: bool = False):
    import glob, os
    found = {os.path.basename(p)[len("asset_"):-len(".png")]: p
             for p in glob.glob("outputs/asset_*.png")}
    if regen or not found:
        print("generating assets...")
        for cls, png in make_assets.remote(VEHICLES).items():
            p = f"outputs/asset_{cls}.png"
            open(p, "wb").write(png)
            print(f"wrote {p}")
        found = {cls: f"outputs/asset_{cls}.png" for cls in VEHICLES}

    assets = {cls: open(p, "rb").read() for cls, p in found.items()}
    out = apply_stickers.remote(open(video_path, "rb").read(), assets)
    open("outputs/sticker.mp4", "wb").write(out)
    print("wrote outputs/sticker.mp4")
