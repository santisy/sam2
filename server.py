import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sys

# SAM2 imports
repo_root = Path(__file__).resolve().parent
os.environ["SAM2_REPO_ROOT"] = str(repo_root)
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

app = FastAPI()

# CORS for local React dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SAM2 setup
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Global state
IMAGE_DIR = None
current_image_path = None
current_image_array = None


class SetDirectoryRequest(BaseModel):
    directory: str


class PredictRequest(BaseModel):
    image_path: str
    points: List[List[float]]  # [[x, y], [x, y], ...]


class CreateMaskRequest(BaseModel):
    image_path: str
    mask: List[List[bool]]
    prompt: str


class UpdateMaskRequest(BaseModel):
    image_path: str
    mask_index: int
    mask: List[List[bool]]
    prompt: str


class DeleteMaskRequest(BaseModel):
    image_path: str
    mask_index: int


class ImageInfo(BaseModel):
    path: str
    filename: str
    mask_count: int


def get_image_hash(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:6]


def get_annotation_path(image_path: Path) -> Path:
    annotations_dir = IMAGE_DIR / "annotations"
    annotations_dir.mkdir(exist_ok=True)
    # Keep same filename structure but in centralized annotations dir
    return annotations_dir / f"{image_path.stem}.json"


@app.post("/api/set-directory")
def set_directory(request: SetDirectoryRequest):
    """Set the working directory for images"""
    global IMAGE_DIR, current_image_path, current_image_array
    
    dir_path = Path(request.directory)
    if not dir_path.exists():
        raise HTTPException(status_code=404, detail="Directory not found")
    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")
    
    IMAGE_DIR = dir_path
    current_image_path = None
    current_image_array = None
    
    return {
        "success": True,
        "directory": str(IMAGE_DIR.absolute())
    }


@app.get("/api/images")
def list_images() -> List[ImageInfo]:
    """List all images in the directory"""
    if IMAGE_DIR is None or not IMAGE_DIR.exists():
        return []
    
    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        for img_path in IMAGE_DIR.rglob(ext):
            if "annotations" in img_path.parts:
                continue
            
            rel_path = str(img_path.relative_to(IMAGE_DIR))
            annotation_path = get_annotation_path(img_path)
            
            mask_count = 0
            if annotation_path.exists():
                with open(annotation_path) as f:
                    data = json.load(f)
                    mask_count = len(data.get("masks", []))
            
            images.append(ImageInfo(
                path=rel_path,
                filename=img_path.name,
                mask_count=mask_count
            ))
    
    return sorted(images, key=lambda x: x.path)


@app.get("/api/image/{path:path}")
def serve_image(path: str):
    """Serve an image file"""
    img_path = IMAGE_DIR / path
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(img_path)


@app.post("/api/predict")
def predict_mask(request: PredictRequest) -> Dict:
    """Run SAM2 inference on points"""
    global current_image_path, current_image_array
    
    img_path = IMAGE_DIR / request.image_path
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Load image if different from current
    if current_image_path != str(img_path):
        image = Image.open(img_path).convert("RGB")
        current_image_array = np.array(image)
        predictor.set_image(current_image_array)
        current_image_path = str(img_path)
    
    # Run prediction
    point_coords = np.array(request.points, dtype=np.float32)
    point_labels = np.ones(len(point_coords), dtype=np.int32)
    
    masks, _, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False,
    )
    
    mask = masks[0].astype(bool)
    
    return {
        "mask": mask.tolist(),
        "shape": mask.shape
    }


@app.post("/api/mask")
def create_mask(request: CreateMaskRequest):
    """Create a new mask for an image"""
    img_path = IMAGE_DIR / request.image_path
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    annotations_dir = IMAGE_DIR / "annotations"
    annotations_dir.mkdir(exist_ok=True)
    
    image_hash = get_image_hash(img_path)
    annotation_path = get_annotation_path(img_path)
    
    if annotation_path.exists():
        with open(annotation_path) as f:
            data = json.load(f)
    else:
        data = {
            "image_filename": img_path.name,
            "image_hash": image_hash,
            "masks": []
        }
    
    mask_array = np.array(request.mask, dtype=bool)
    mask_img = Image.fromarray((mask_array.astype(np.uint8)) * 255)
    
    mask_index = len(data["masks"]) + 1
    mask_filename = f"{img_path.stem}-{image_hash}-mask-{mask_index:03d}.png"
    mask_path = annotations_dir / mask_filename
    mask_img.save(mask_path)
    
    data["masks"].append({
        "mask_filename": mask_filename,
        "prompt": request.prompt
    })
    
    with open(annotation_path, "w") as f:
        json.dump(data, f, indent=2)
    
    return {"success": True, "mask_index": mask_index - 1}


@app.put("/api/mask")
def update_mask(request: UpdateMaskRequest):
    """Update an existing mask"""
    img_path = IMAGE_DIR / request.image_path
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    annotation_path = get_annotation_path(img_path)
    if not annotation_path.exists():
        raise HTTPException(status_code=404, detail="No annotations found")
    
    with open(annotation_path) as f:
        data = json.load(f)
    
    if request.mask_index >= len(data["masks"]):
        raise HTTPException(status_code=404, detail="Mask index out of range")
    
    annotations_dir = IMAGE_DIR / "annotations"
    mask_filename = data["masks"][request.mask_index]["mask_filename"]
    mask_path = annotations_dir / mask_filename
    
    mask_array = np.array(request.mask, dtype=bool)
    mask_img = Image.fromarray((mask_array.astype(np.uint8)) * 255)
    mask_img.save(mask_path)
    
    data["masks"][request.mask_index]["prompt"] = request.prompt
    
    with open(annotation_path, "w") as f:
        json.dump(data, f, indent=2)
    
    return {"success": True}


@app.delete("/api/mask")
def delete_mask(request: DeleteMaskRequest):
    """Delete a mask"""
    img_path = IMAGE_DIR / request.image_path
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    annotation_path = get_annotation_path(img_path)
    if not annotation_path.exists():
        raise HTTPException(status_code=404, detail="No annotations found")
    
    with open(annotation_path) as f:
        data = json.load(f)
    
    if request.mask_index >= len(data["masks"]):
        raise HTTPException(status_code=404, detail="Mask index out of range")
    
    annotations_dir = IMAGE_DIR / "annotations"
    mask_filename = data["masks"][request.mask_index]["mask_filename"]
    mask_path = annotations_dir / mask_filename
    
    if mask_path.exists():
        os.remove(mask_path)
    
    data["masks"].pop(request.mask_index)
    
    with open(annotation_path, "w") as f:
        json.dump(data, f, indent=2)
    
    return {"success": True}


@app.get("/api/annotation/{path:path}")
def get_annotation(path: str):
    """Get annotation for an image"""
    img_path = IMAGE_DIR / path
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    annotation_path = get_annotation_path(img_path)
    if not annotation_path.exists():
        return {"masks": []}
    
    with open(annotation_path) as f:
        data = json.load(f)
    
    # Load mask images and convert to boolean arrays
    annotations_dir = IMAGE_DIR / "annotations"
    for mask_info in data.get("masks", []):
        mask_path = annotations_dir / mask_info["mask_filename"]
        if mask_path.exists():
            mask_img = Image.open(mask_path)
            mask_array = np.array(mask_img) > 127
            mask_info["mask_data"] = mask_array.tolist()
    
    return data


@app.get("/api/config")
def get_config():
    """Get server configuration"""
    return {
        "image_dir": str(IMAGE_DIR.absolute()),
        "total_images": len(list_images())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=5876, reload=True, log_level="debug")