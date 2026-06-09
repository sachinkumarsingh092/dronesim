import modal

app = modal.App("dronesim-cartoon")
vol = modal.Volume.from_name("hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision")
    .pip_install("diffusers", "transformers", "accelerate",
                 "safetensors", "opencv-python-headless", "numpy", "pillow")
)

@app.function(gpu="A100", image=image, timeout=2400,
              volumes={"/root/.cache/huggingface": vol})
def cartoonize(video_bytes: bytes, prompt: str) -> bytes:
    import numpy as np, cv2, torch
    from PIL import Image
    from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel

    open("/tmp/in.mp4", "wb").write(video_bytes)
    cap, frames = cv2.VideoCapture("/tmp/in.mp4"), []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    cap.release()
    h, w = frames[0].shape[:2]

    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained("Lykon/dreamshaper-8", controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None).to("cuda")

    W, H = 512, 288
    neg = "blurry, photo, realistic, low quality, deformed"
    vw = cv2.VideoWriter("/tmp/out.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 6, (w, h))
    for f in frames:
        small = cv2.resize(f, (W, H))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        ctrl = Image.fromarray(cv2.cvtColor(cv2.Canny(gray, 80, 160), cv2.COLOR_GRAY2RGB))
        init = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        g = torch.Generator("cuda").manual_seed(0)
        out = pipe(prompt=prompt, negative_prompt=neg, image=init, control_image=ctrl,
                   strength=0.6, num_inference_steps=20, guidance_scale=7.0,
                   controlnet_conditioning_scale=0.9, generator=g).images[0]
        res = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
        vw.write(cv2.resize(res, (w, h)))
    vw.release()
    return open("/tmp/out.mp4", "rb").read()

@app.local_entrypoint()
def main(video_path: str, prompt: str = "storybook cartoon illustration, whimsical enchanted village, vibrant hand-painted, studio ghibli style"):
    from pathlib import Path
    out = f"outputs/{Path(__file__).stem}.mp4"
    data = open(video_path, "rb").read()
    open(out, "wb").write(cartoonize.remote(data, prompt))
    print(f"wrote {out}")

