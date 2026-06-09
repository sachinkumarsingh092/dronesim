import modal

app = modal.App("dronesim-sam")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "torchvision")
    .env({"SAM2_BUILD_CUDA": "0"})  # skip CUDA ext build → reliable install
    .pip_install(
        "opencv-python-headless", "numpy", "huggingface_hub",
        "transformers", "pillow",
        "git+https://github.com/facebookresearch/sam2.git",
    )
)

@app.function(gpu="T4", image=image, timeout=900)
def segment(img_bytes: bytes) -> bytes:
    import numpy as np, cv2
    from sam2.build_sam import build_sam2_hf
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    model = build_sam2_hf("facebook/sam2-hiera-tiny")
    masks = SAM2AutomaticMaskGenerator(model).generate(rgb)

    overlay = img.copy()
    rng = np.random.default_rng(0)
    for m in masks:
        overlay[m["segmentation"]] = rng.integers(0, 255, 3)
    out = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
    return cv2.imencode(".png", out)[1].tobytes()

@app.local_entrypoint()
def main(image_path: str):
    data = open(image_path, "rb").read()
    open("sam_out.png", "wb").write(segment.remote(data))
    print("wrote sam_out.png")

