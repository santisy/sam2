import os
from pathlib import Path
import numpy as np
from PIL import Image
import sys

# SAM2 imports
repo_root = Path(__file__).resolve().parent.parent
os.environ["SAM2_REPO_ROOT"] = str(repo_root)
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Model weights (lazy loaded on first predict call)
_predictor = None
_current_image_path = None
_current_image_array = None

def get_predictor():
    global _predictor
    if _predictor is None:
        print("Loading SAM2 model weights...")
        checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        _predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
        print("SAM2 model loaded!")
    return _predictor

def predict_mask(image_path, points):
    global _current_image_path, _current_image_array
    
    predictor = get_predictor()
    
    # Load image if different from current
    if _current_image_path != str(image_path):
        image = Image.open(image_path).convert("RGB")
        _current_image_array = np.array(image)
        predictor.set_image(_current_image_array)
        _current_image_path = str(image_path)
    
    # Run prediction
    point_coords = np.array(points, dtype=np.float32)
    point_labels = np.ones(len(point_coords), dtype=np.int32)
    
    masks, _, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False,
    )
    
    mask = masks[0].astype(bool)
    return mask