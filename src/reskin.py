import modal

app = modal.App("dronesim-reskin")
vol = modal.Volume.from_name("hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "torchvision")
    .env({"SAM2_BUILD_CUDA": "0"})
    .pip_install(
        "diffusers", "transformers", "accelerate", "safetensors",
        "opencv-python-headless", "numpy", "pillow", "huggingface_hub",
        "git+https://github.com/facebookresearch/sam2.git",
    )
)

@app.function(gpu="A100", image=image, timeout=3600,
              volumes={"/root/.cache/huggingface": vol})
def reskin(video_bytes: bytes, prompt: str, det_prompt: str) -> bytes:
    import numpy as np, cv2, torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel

    open("/tmp/in.mp4", "wb").write(video_bytes)
    cap, frames = cv2.VideoCapture("/tmp/in.mp4"), []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    cap.release()
    h, w = frames[0].shape[:2]

    proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    gd = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to("cuda")
    sam = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained("Lykon/dreamshaper-8", controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None).to("cuda")

    W, H, SCALE = 512, 288, 3
    neg = "blurry, photo, realistic, low quality, deformed"
    vw = cv2.VideoWriter("/tmp/out.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 6, (w, h))
    for f in frames:
        # detect (upscaled) -> boxes
        big = cv2.resize(f, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
        pil_big = Image.fromarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB))
        inp = proc(images=pil_big, text=det_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = gd(**inp)
        res = proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=0.15, text_threshold=0.15,
            target_sizes=[pil_big.size[::-1]])[0]
        boxes = res["boxes"].cpu().numpy() / SCALE

        # union mask of all objects (full res)
        mask = np.zeros((h, w), bool)
        if len(boxes):
            sam.set_image(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            for box in boxes:
                m, _, _ = sam.predict(box=box[None, :], multimask_output=False)
                mask |= m[0].astype(bool)

        out_frame = f.copy()
        if mask.any():
            small = cv2.resize(f, (W, H))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            ctrl = Image.fromarray(cv2.cvtColor(cv2.Canny(gray, 80, 160), cv2.COLOR_GRAY2RGB))
            init = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            g = torch.Generator("cuda").manual_seed(0)
            styl = pipe(prompt=prompt, negative_prompt=neg, image=init, control_image=ctrl,
                        strength=0.7, num_inference_steps=20, guidance_scale=7.0,
                        controlnet_conditioning_scale=0.9, generator=g).images[0]
            styl = cv2.resize(cv2.cvtColor(np.array(styl), cv2.COLOR_RGB2BGR), (w, h))
            # composite ONLY masked pixels (feathered edge)
            m3 = cv2.GaussianBlur((mask * 255).astype(np.uint8), (5, 5), 0)[..., None] / 255.0
            out_frame = (styl * m3 + f * (1 - m3)).astype(np.uint8)
        vw.write(out_frame)
    vw.release()
    return open("/tmp/out.mp4", "rb").read()

@app.local_entrypoint()
def main(video_path: str, prompt: str = "cute cartoon toy vehicle, storybook illustration, hand-painted, vibrant", det_prompt: str = "car . van . truck . bus ."):
    from pathlib import Path
    out = f"outputs/{Path(__file__).stem}.mp4"
    data = open(video_path, "rb").read()
    open(out, "wb").write(reskin.remote(data, prompt, det_prompt))
    print(f"wrote {out}")

