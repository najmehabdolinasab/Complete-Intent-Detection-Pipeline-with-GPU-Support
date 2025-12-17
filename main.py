from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from setfit import SetFitModel
import torch
import time

# تنظیمات سخت‌افزاری
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "model/persian_intent_model"

app = FastAPI(title="Persian Intent Detection Service")

# لود کردن مدل در حافظه (Singleton Pattern)
# این کار باعث می‌شود مدل فقط یکبار در VRAM لود شود
try:
    print(f"🚀 Loading model to {device}...")
    model = SetFitModel.from_pretrained(MODEL_PATH).to(device)
    print("✅ Model is ready for inference.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

class Query(BaseModel):
    text: str

@app.get("/health")
def health_check():
    return {"status": "up", "device": device, "model_loaded": model is not None}

@app.post("/predict")
async def predict(data: Query):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.perf_counter()
        
        # استنتاج روی GPU بدون محاسبه گرادینت برای سرعت بیشتر
        with torch.no_grad():
            intent = model(data.text)
            
        end_time = time.perf_counter()
        latency = (end_time - start_time) * 1000
        
        return {
            "intent": str(intent),
            "latency_ms": round(latency, 2),
            "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8025)