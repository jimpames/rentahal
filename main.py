# this is the fastapi server for the llama and llava AI worker nodes 

import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import json
import base64
from io import BytesIO
from PIL import Image
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app_llama = FastAPI()
app_llava = FastAPI()

class Input(BaseModel):
    prompt: str

class ImageInput(BaseModel):
    prompt: str
    image: str  # Base64 encoded image

def ollama_predict(prompt: str, model: str) -> str:
    try:
        result = subprocess.run(
            ["ollama", "run", f"{model}:latest", prompt],
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )
        filtered_output = "\n".join(
            line for line in result.stdout.splitlines()
            if "failed to get console mode" not in line
        )
        try:
            output = json.loads(filtered_output)
            return output.get('text', 'No prediction available')
        except json.JSONDecodeError:
            return filtered_output.strip()
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

@app_llama.post("/predict/")
async def predict_llama(input: Input):
    response = ollama_predict(input.prompt, "llama3")
    return {"response": response}

@app_llava.post("/predict/")
async def predict_llava(input: ImageInput):
    temp_image_path = "temp_image.jpg"
    try:
        logger.debug(f"Received prompt: {input.prompt}")
        image_data = base64.b64decode(input.image)
        image = Image.open(BytesIO(image_data))
        
        image.save(temp_image_path)
        logger.debug(f"Saved temporary image to {temp_image_path}")
        
        command = [
            "ollama", "run", "llava",
            f"{input.prompt} Describe this image: ./{temp_image_path}"
        ]
        logger.debug(f"Executing command: {' '.join(command)}")
        
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        
        filtered_output = "\n".join(
            line for line in result.stdout.splitlines()
            if "failed to get console mode" not in line
        )
        
        try:
            output = json.loads(filtered_output)
            response = output.get('text', 'No prediction available')
        except json.JSONDecodeError:
            response = filtered_output.strip()
        
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLaVA prediction failed: {str(e)}")
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

@app_llama.get("/health")
@app_llava.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    from multiprocessing import Process

    def run_llama():
        uvicorn.run(app_llama, host="0.0.0.0", port=8000)

    def run_llava():
        uvicorn.run(app_llava, host="0.0.0.0", port=8001)

    llama_process = Process(target=run_llama)
    llava_process = Process(target=run_llava)

    llama_process.start()
    llava_process.start()

    llama_process.join()
    llava_process.join()
