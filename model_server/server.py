import base64
import io
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from diffusers import StableDiffusionXLPipeline

app = FastAPI(title="Lexaday API Server")

# Read default parameter values from environment variables, if set
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("DEFAULT_MAX_NEW_TOKENS", 80))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", 0.7))
DEFAULT_DO_SAMPLE = os.getenv("DEFAULT_DO_SAMPLE", "true").lower() in ("true", "1", "yes")
DEFAULT_NUM_INFERENCE_STEPS = int(os.getenv("DEFAULT_NUM_INFERENCE_STEPS", 30))
DEFAULT_GUIDANCE_SCALE = float(os.getenv("DEFAULT_GUIDANCE_SCALE", 7.0))

# -------------------------------
# Request Models
# -------------------------------
class TextRequest(BaseModel):
    prompt: str
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    do_sample: bool = DEFAULT_DO_SAMPLE

class ImageRequest(BaseModel):
    prompt: str
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE

# -------------------------------
# Load Qwen Model for Text Generation
# -------------------------------
print("🔄 Loading Qwen model...")
qwen_model_id = "Qwen/Qwen1.5-1.8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(qwen_model_id)
qwen_model = AutoModelForCausalLM.from_pretrained(
    qwen_model_id,
    device_map="auto",
    torch_dtype="auto"
)
text_generator = pipeline("text-generation", model=qwen_model, tokenizer=tokenizer)
print("✅ Qwen model loaded.")

# -------------------------------
# Load Stable Diffusion XL Model for Image Generation
# -------------------------------
print("🎨 Loading Stable Diffusion XL model...")
sdxl_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
    sdxl_model_id,
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")
print("✅ SDXL model loaded.")

# -------------------------------
# API Endpoints
# -------------------------------
@app.post("/generate-text")
async def generate_text(req: TextRequest):
    try:
        outputs = text_generator(
            req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            do_sample=req.do_sample,
            return_full_text=False
        )
        generated_text = outputs[0]['generated_text']
        return {"quote": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-image")
async def generate_image(req: ImageRequest):
    try:
        result = sdxl_pipe(req.prompt, num_inference_steps=req.num_inference_steps, guidance_scale=req.guidance_scale)
        image = result.images[0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {"image_base64": img_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Lexaday API is running."}