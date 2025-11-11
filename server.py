import os
import json
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# CONFIG
PORT = 5876
ENABLE_SAM = False  # False = lazy load on first call, True = load on startup

from services import sam_service
sam_service.ENABLE_SAM = ENABLE_SAM

if ENABLE_SAM:
    sam_service.load_weights()

app = FastAPI()

# CORS for local React dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
IMAGE_DIR = None


def get_image_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:6]


def get_annotation_path(image_path):
    annotations_dir = IMAGE_DIR / "annotations"
    annotations_dir.mkdir(exist_ok=True)
    return annotations_dir / f"{image_path.stem}.json"


@app.post("/api/set-directory")
async def set_directory(request: dict):
    global IMAGE_DIR
    
    dir_path = Path(request['directory'])
    if not dir_path.exists():
        raise HTTPException(status_code=404, detail="Directory not found")
    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")
    
    IMAGE_DIR = dir_path
    
    return {
        "success": True,
        "directory": str(IMAGE_DIR.absolute())
    }


@app.get("/api/images")
def list_images():
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
            
            images.append({
                "path": rel_path,
                "filename": img_path.name,
                "mask_count": mask_count
            })
    
    return sorted(images, key=lambda x: x['path'])


@app.get("/api/image/{path:path}")
def serve_image(path):
    img_path = IMAGE_DIR / path
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(img_path)


@app.post("/api/predict")
async def predict_mask(request: dict):
    img_path = IMAGE_DIR / request['image_path']
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    mask = sam_service.predict_mask(img_path, request['points'])
    
    return {
        "mask": mask.tolist(),
        "shape": mask.shape
    }


@app.post("/api/mask")
async def create_mask(request: dict):
    img_path = IMAGE_DIR / request['image_path']
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
    
    mask_array = np.array(request['mask'], dtype=bool)
    mask_img = Image.fromarray((mask_array.astype(np.uint8)) * 255)
    
    mask_index = len(data["masks"]) + 1
    mask_filename = f"{img_path.stem}-{image_hash}-mask-{mask_index:03d}.png"
    mask_path = annotations_dir / mask_filename
    mask_img.save(mask_path)
    
    data["masks"].append({
        "mask_filename": mask_filename,
        "prompt": request['prompt']
    })
    
    with open(annotation_path, "w") as f:
        json.dump(data, f, indent=2)
    
    return {"success": True, "mask_index": mask_index - 1}


@app.put("/api/mask")
async def update_mask(request: dict):
    img_path = IMAGE_DIR / request['image_path']
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    annotation_path = get_annotation_path(img_path)
    if not annotation_path.exists():
        raise HTTPException(status_code=404, detail="No annotations found")
    
    with open(annotation_path) as f:
        data = json.load(f)
    
    mask_index = request['mask_index']
    if mask_index >= len(data["masks"]):
        raise HTTPException(status_code=404, detail="Mask index out of range")
    
    annotations_dir = IMAGE_DIR / "annotations"
    mask_filename = data["masks"][mask_index]["mask_filename"]
    mask_path = annotations_dir / mask_filename
    
    mask_array = np.array(request['mask'], dtype=bool)
    mask_img = Image.fromarray((mask_array.astype(np.uint8)) * 255)
    mask_img.save(mask_path)
    
    data["masks"][mask_index]["prompt"] = request['prompt']
    
    with open(annotation_path, "w") as f:
        json.dump(data, f, indent=2)
    
    return {"success": True}


@app.delete("/api/mask")
async def delete_mask(request: dict):
    img_path = IMAGE_DIR / request['image_path']
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    annotation_path = get_annotation_path(img_path)
    if not annotation_path.exists():
        raise HTTPException(status_code=404, detail="No annotations found")
    
    with open(annotation_path) as f:
        data = json.load(f)
    
    mask_index = request['mask_index']
    if mask_index >= len(data["masks"]):
        raise HTTPException(status_code=404, detail="Mask index out of range")
    
    annotations_dir = IMAGE_DIR / "annotations"
    mask_filename = data["masks"][mask_index]["mask_filename"]
    mask_path = annotations_dir / mask_filename
    
    if mask_path.exists():
        os.remove(mask_path)
    
    data["masks"].pop(mask_index)
    
    with open(annotation_path, "w") as f:
        json.dump(data, f, indent=2)
    
    return {"success": True}


@app.get("/api/annotation/{path:path}")
def get_annotation(path):
    img_path = IMAGE_DIR / path
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    annotation_path = get_annotation_path(img_path)
    if not annotation_path.exists():
        return {"masks": []}
    
    with open(annotation_path) as f:
        data = json.load(f)
    
    annotations_dir = IMAGE_DIR / "annotations"
    for mask_info in data.get("masks", []):
        mask_path = annotations_dir / mask_info["mask_filename"]
        if mask_path.exists():
            mask_img = Image.open(mask_path)
            mask_array = np.array(mask_img) > 127
            mask_info["mask_data"] = mask_array.tolist()
    
    return data


@app.post("/api/crop")
async def save_crop(request: dict):
    img_path = IMAGE_DIR / request['image_path']
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    annotation_path = get_annotation_path(img_path)
    image_hash = get_image_hash(img_path)
    
    if annotation_path.exists():
        with open(annotation_path) as f:
            data = json.load(f)
        
        # Delete old cropped image if updating
        if "crop" in data and "crop_filename" in data["crop"]:
            crops_dir = IMAGE_DIR / "annotations" / "crops"
            old_crop_path = crops_dir / data["crop"]["crop_filename"]
            if old_crop_path.exists():
                os.remove(old_crop_path)
    else:
        annotations_dir = IMAGE_DIR / "annotations"
        annotations_dir.mkdir(exist_ok=True)
        data = {
            "image_filename": img_path.name,
            "image_hash": image_hash,
            "masks": []
        }
    
    crop_info = request['crop']
    
    # Save cropped image
    crops_dir = IMAGE_DIR / "annotations" / "crops"
    crops_dir.mkdir(exist_ok=True)
    
    image = Image.open(img_path)
    x = int(crop_info['x'])
    y = int(crop_info['y'])
    width = int(crop_info['width'])
    height = int(crop_info['height'])
    
    cropped = image.crop((x, y, x + width, y + height))
    
    crop_filename = f"{img_path.stem}-{image_hash}-crop.png"
    crop_path = crops_dir / crop_filename
    cropped.save(crop_path, "PNG")
    
    crop_info['crop_filename'] = crop_filename
    data["crop"] = crop_info
    
    with open(annotation_path, "w") as f:
        json.dump(data, f, indent=2)
    
    return {"success": True}


@app.delete("/api/crop")
async def delete_crop(request: dict):
    img_path = IMAGE_DIR / request['image_path']
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    annotation_path = get_annotation_path(img_path)
    if not annotation_path.exists():
        return {"success": True}
    
    with open(annotation_path) as f:
        data = json.load(f)
    
    if "crop" in data:
        # Delete cropped image file if it exists
        if "crop_filename" in data["crop"]:
            crops_dir = IMAGE_DIR / "annotations" / "crops"
            crop_path = crops_dir / data["crop"]["crop_filename"]
            if crop_path.exists():
                os.remove(crop_path)
        
        del data["crop"]
        
        with open(annotation_path, "w") as f:
            json.dump(data, f, indent=2)
    
    return {"success": True}


@app.get("/api/crop-image/{filename}")
def serve_crop_image(filename):
    crops_dir = IMAGE_DIR / "annotations" / "crops"
    crop_path = crops_dir / filename
    if not crop_path.exists():
        raise HTTPException(status_code=404, detail="Cropped image not found")
    return FileResponse(crop_path)


@app.get("/api/config")
def get_config():
    return {
        "image_dir": str(IMAGE_DIR.absolute()),
        "total_images": len(list_images())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, reload=True, log_level="debug")