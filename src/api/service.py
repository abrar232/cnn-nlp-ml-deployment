import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import io
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from src.utils.preprocessing import preprocess_image
from src.model_training.model import PlantCNN
import mlflow.pytorch

# ── Class names ───────────────────────────────────────────────
CLASS_NAMES = [
    'Loose Silky-bent', 'Charlock', 'Cleavers', 'Common Chickweed',
    'Common wheat', 'Fat Hen', 'Black-grass', 'Maize',
    'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill',
    'Sugar beet'
]

# ── Load model once at startup ────────────────────────────────
mlflow.set_tracking_uri("sqlite:///mlflow.db")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if os.environ.get('RUNNING_IN_DOCKER'):
    # Inside Docker — load directly from file
    from src.model_training.model import PlantCNN
    model = PlantCNN(num_classes=12).to(device)
    model.load_state_dict(torch.load('src/model/plant_cnn.pth', map_location=device))
else:
    # Running locally — load from MLflow registry
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model = mlflow.pytorch.load_model("models:/plant-cnn@production")
    model = model.to(device)

model.eval()

# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(title='Plant Seedling Classifier')


@app.get('/health')
def health():
    return {'status': 'ok', 'model': 'PlantCNN', 'classes': len(CLASS_NAMES)}


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # 1. Validate it's an image
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='File must be an image')

    # 2. Read the uploaded bytes and open as Pillow image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # 3. Preprocess using our utils function
    img_tensor = preprocess_image(image).to(device)

    # 4. Run prediction
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = probs.argmax(1).item()
        confidence = probs[0][pred_idx].item() * 100

    return {
        'prediction': CLASS_NAMES[pred_idx],
        'confidence': round(confidence, 2),
        'class_index': pred_idx
    }