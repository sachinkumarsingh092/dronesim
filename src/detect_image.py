import modal

app = modal.App("dronesim-det")

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

@app.function(gpu="T4", image=image, timeout=900)
def detect_segment(img_bytes: bytes, prompt: str) -> bytes:
    import numpy as np, cv2, torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    gd = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to("cuda")

    inputs = proc(images=pil, text=prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = gd(**inputs)
    res = proc.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=0.25, text_threshold=0.25,
        target_sizes=[pil.size[::-1]])[0]
    boxes = res["boxes"].cpu().numpy()
    labels = res.get("text_labels", res.get("labels"))

    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")
    predictor.set_image(rgb)
    overlay = img.copy()
    rng = np.random.default_rng(0)
    for box, label in zip(boxes, labels):
        masks, _, _ = predictor.predict(box=box[None, :], multimask_output=False)
        overlay[masks[0].astype(bool)] = rng.integers(0, 255, 3)
        cv2.putText(overlay, label, (int(box[0]), max(int(box[1]) - 3, 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    out = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
    return cv2.imencode(".png", out)[1].tobytes()

@app.local_entrypoint()
def main(image_path: str, prompt: str = "car. person. bus. truck."):
    data = open(image_path, "rb").read()
    open("det_out.png", "wb").write(detect_segment.remote(data, prompt))
    print("wrote det_out.png")

