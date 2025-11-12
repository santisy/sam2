import os
import json
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
IMAGE_DIR = Path("test_data_total_1029")  # Hardcoded to match frontend default


from services import mask_crop_service as mcs
from services import video_gen_service as vgs


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
            annotation_path = mcs.get_annotation_path(IMAGE_DIR, img_path)
            
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
    
    mask_array = np.array(request['mask'], dtype=bool)
    mask_index = mcs.create_mask(IMAGE_DIR, img_path, mask_array, request['prompt'])
    
    return {"success": True, "mask_index": mask_index}


@app.put("/api/mask")
async def update_mask(request: dict):
    img_path = IMAGE_DIR / request['image_path']
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    mask_array = np.array(request['mask'], dtype=bool)
    success = mcs.update_mask(IMAGE_DIR, img_path, request['mask_index'], mask_array, request['prompt'])
    
    if not success:
        raise HTTPException(status_code=404, detail="Mask not found or invalid index")
    
    return {"success": True}


@app.delete("/api/mask")
async def delete_mask(request: dict):
    img_path = IMAGE_DIR / request['image_path']
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    success = mcs.delete_mask(IMAGE_DIR, img_path, request['mask_index'])
    
    if not success:
        raise HTTPException(status_code=404, detail="Mask not found or invalid index")
    
    return {"success": True}


@app.get("/api/annotation/{path:path}")
def get_annotation(path):
    img_path = IMAGE_DIR / path
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    return mcs.get_annotation_with_masks(IMAGE_DIR, img_path)


@app.post("/api/crop")
async def save_crop(request: dict):
    img_path = IMAGE_DIR / request['image_path']
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    mcs.save_crop(IMAGE_DIR, img_path, request['crop'])
    
    return {"success": True}


@app.delete("/api/crop")
async def delete_crop(request: dict):
    img_path = IMAGE_DIR / request['image_path']
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    mcs.delete_crop(IMAGE_DIR, img_path)
    
    return {"success": True}


@app.get("/api/crop-image/{filename}")
def serve_crop_image(filename):
    crops_dir = IMAGE_DIR / "annotations" / "crops"
    crop_path = crops_dir / filename
    if not crop_path.exists():
        raise HTTPException(status_code=404, detail="Cropped image not found")
    return FileResponse(crop_path)


@app.post("/api/generate/sora2")
async def generate_sora2(request: dict):
    img_path = IMAGE_DIR / request['image_path']
    annotation_path = mcs.get_annotation_path(IMAGE_DIR, img_path)
    
    with open(annotation_path) as f:
        data = json.load(f)
    
    mask_index = request['mask_index']
    crops_dir = IMAGE_DIR / "annotations" / "crops"
    crop_path = crops_dir / data["crop"]["crop_filename"]
    
    job_id = vgs.submit_job(
        image_dir=IMAGE_DIR,
        image_path=request['image_path'],
        mask_index=mask_index,
        image_hash=data["image_hash"],
        crop_path=crop_path,
        mask_prompt=data["masks"][mask_index]["prompt"],
        aspect_ratio=data["crop"]["orientation"],
        model='sora2'
    )
    
    return {"success": True, "job_id": job_id}


@app.post("/api/generate/veo3")
async def generate_veo3(request: dict):
    img_path = IMAGE_DIR / request['image_path']
    annotation_path = mcs.get_annotation_path(IMAGE_DIR, img_path)
    
    with open(annotation_path) as f:
        data = json.load(f)
    
    mask_index = request['mask_index']
    crops_dir = IMAGE_DIR / "annotations" / "crops"
    crop_path = crops_dir / data["crop"]["crop_filename"]
    
    job_id = vgs.submit_job(
        image_dir=IMAGE_DIR,
        image_path=request['image_path'],
        mask_index=mask_index,
        image_hash=data["image_hash"],
        crop_path=crop_path,
        mask_prompt=data["masks"][mask_index]["prompt"],
        aspect_ratio=data["crop"]["orientation"],
        model='veo3'
    )
    
    return {"success": True, "job_id": job_id}


@app.get("/api/generations")
def list_generations():
    return vgs.list_all_jobs()


@app.get("/api/config")
def get_config():
    return {
        "image_dir": str(IMAGE_DIR.absolute()),
        "total_images": len(list_images())
    }


@app.get("/api/video/ours/{filename}")
def serve_our_video(filename):
    """Serve video from annotations/ours directory"""
    ours_dir = IMAGE_DIR / "annotations" / "ours"
    video_path = ours_dir / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_path)


@app.get("/api/video/sora2/{filename}")
def serve_sora2_video(filename):
    """Serve video from annotations/sora2 directory"""
    sora2_dir = IMAGE_DIR / "annotations" / "sora2"
    video_path = sora2_dir / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_path)


@app.get("/api/video/veo3/{filename}")
def serve_veo3_video(filename):
    """Serve video from annotations/veo3 directory"""
    veo3_dir = IMAGE_DIR / "annotations" / "veo3"
    video_path = veo3_dir / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_path)


@app.on_event("startup")
async def startup_event():
    if IMAGE_DIR and IMAGE_DIR.exists():
        vgs.load_all_jobs(IMAGE_DIR)
    else:
        print(f"Image directory not found: {IMAGE_DIR}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, reload=True, log_level="debug")