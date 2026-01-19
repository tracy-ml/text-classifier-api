from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import os

app = FastAPI(title="M2 Text Classifier API")

# Load transformer model on startup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
MODEL_PATH = os.path.join(project_root, "models", "transformer")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

print("Loading transformer model...")
classifier = pipeline("text-classification", model=MODEL_PATH, tokenizer=MODEL_PATH)

# AG News label mapping (pipeline uses LABEL_0, LABEL_1, etc.)
LABELS = {
    'LABEL_0': "World",
    'LABEL_1': "Sports", 
    'LABEL_2': "Business",
    'LABEL_3': "Sci/Tech"
}

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    probabilities: dict[str, float]
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest) -> PredictResponse:
    # Get prediction
    result = classifier(req.text, return_all_scores=True)[0]
    
    # Find the highest score
    best_result = max(result, key=lambda x: x['score'])
    label = LABELS[best_result['label']]
    
    # Create probabilities dict
    probabilities = {LABELS[r['label']]: float(r['score']) for r in result}
    
    return PredictResponse(
        label=label,
        probabilities=probabilities,
        text=req.text
    )
