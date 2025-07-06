from flask import Flask, request, jsonify
from diffusers import StableDiffusionPipeline
import torch
import base64
from io import BytesIO

app = Flask(__name__)

# Load Stable Diffusion model (runs on GPU or CPU)
model_id = "runwayml/stable-diffusion-v1-5"
try:
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")  # GPU ke liye; CPU ke liye comment out karen
except:
    pipe = StableDiffusionPipeline.from_pretrained(model_id)  # CPU fallback

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        negative_prompt = data.get('negative_prompt', '')
        
        # Generate image
        image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=30).images[0]
        
        # Convert image to base64 for frontend
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({"status": "success", "image": f"data:image/png;base64,{img_str}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
