import torch
import numpy as np
from PIL import Image

def preprocess_image(image: Image.Image) -> torch.Tensor:
    # 1. Resize to what the model was trained on
    image = image.convert('RGB')
    image = image.resize((128, 128))
    
    # 2. Convert to numpy and normalise to 0-1
    img_array = np.array(image).astype('float32') / 255.0
    
    # 3. Convert to tensor and reshape for PyTorch
    # NumPy is (H, W, C) but PyTorch expects (C, H, W)
    img_tensor = torch.tensor(img_array).permute(2, 0, 1)
    
    # 4. Add batch dimension — model expects (N, C, H, W)
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor