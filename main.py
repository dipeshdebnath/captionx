from flask import Flask, request, jsonify
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoImageProcessor
from PIL import Image
import torch
import io

# Initialize model and processors once
model_id = "cnmoro/tiny-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
image_processor = AutoImageProcessor.from_pretrained(model_id)

app = Flask(__name__)

def generate_caption(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, num_beams=3)
    caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return caption

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "🖼️ Tiny Image Captioning API is running.",
        "model": model_id,
        "endpoints": {
            "POST /caption": "Upload an image to receive a caption."
        }
    })

@app.route("/caption", methods=["POST"])
def caption_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()

    try:
        caption = generate_caption(image_bytes)
        return jsonify({"caption": caption})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
