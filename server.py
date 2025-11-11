import os
import json
import hashlib
import base64
from pathlib import Path
from datetime import datetime
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


@app.post("/api/generate/sora2")
async def generate_sora2(request: dict):
    from services import sora2
    
    img_path = IMAGE_DIR / request['image_path']
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    annotation_path = get_annotation_path(img_path)
    if not annotation_path.exists():
        raise HTTPException(status_code=404, detail="No annotations found")
    
    with open(annotation_path) as f:
        data = json.load(f)
    
    if "crop" not in data:
        raise HTTPException(status_code=400, detail="No crop found for this image")
    
    mask_index = request['mask_index']
    if mask_index >= len(data.get("masks", [])):
        raise HTTPException(status_code=404, detail="Mask index out of range")
    
    # Get crop image path and prompt
    crops_dir = IMAGE_DIR / "annotations" / "crops"
    crop_filename = data["crop"]["crop_filename"]
    crop_path = crops_dir / crop_filename
    
    if not crop_path.exists():
        raise HTTPException(status_code=404, detail="Crop image not found")
    
    mask_prompt = data["masks"][mask_index]["prompt"]
    aspect_ratio = data["crop"]["orientation"]
    image_hash = data["image_hash"]
    
    # Submit to Sora2
    result = sora2.submit_i2v(
        image_path=str(crop_path),
        prompt=mask_prompt,
        aspect_ratio=aspect_ratio
    )
    
    # Create sora2 directory
    sora2_dir = IMAGE_DIR / "annotations" / "sora2"
    sora2_dir.mkdir(exist_ok=True)
    
    # Use mask-based JSON naming
    mask_json_filename = f"{img_path.stem}-{image_hash}-mask-{mask_index + 1:03d}.json"
    mask_json_path = sora2_dir / mask_json_filename
    
    # Load or create mask JSON
    if mask_json_path.exists():
        with open(mask_json_path) as f:
            mask_data = json.load(f)
    else:
        mask_data = {
            "image_path": request['image_path'],
            "mask_index": mask_index,
            "jobs": []
        }
    
    # Add new job - Sora2 only needs job_id
    job = {
        "job_id": result['job_id'],
        "model": "sora2",
        "prompt": mask_prompt,
        "crop_filename": crop_filename,
        "aspect_ratio": aspect_ratio,
        "duration": 4,
        "status": result['status'],
        "created_at": datetime.now().isoformat()
    }
    
    mask_data["jobs"].append(job)
    
    with open(mask_json_path, "w") as f:
        json.dump(mask_data, f, indent=2)
    
    # Start polling for all incomplete jobs
    start_polling_all_jobs()
    
    return {
        "success": True,
        "job_id": result['job_id'],
        "status": result['status']
    }


@app.post("/api/generate/veo3")
async def generate_veo3(request: dict):
    from services import veo3
    
    img_path = IMAGE_DIR / request['image_path']
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    annotation_path = get_annotation_path(img_path)
    if not annotation_path.exists():
        raise HTTPException(status_code=404, detail="No annotations found")
    
    with open(annotation_path) as f:
        data = json.load(f)
    
    if "crop" not in data:
        raise HTTPException(status_code=400, detail="No crop found for this image")
    
    mask_index = request['mask_index']
    if mask_index >= len(data.get("masks", [])):
        raise HTTPException(status_code=404, detail="Mask index out of range")
    
    # Get crop image path and prompt
    crops_dir = IMAGE_DIR / "annotations" / "crops"
    crop_filename = data["crop"]["crop_filename"]
    crop_path = crops_dir / crop_filename
    
    if not crop_path.exists():
        raise HTTPException(status_code=404, detail="Crop image not found")
    
    mask_prompt = data["masks"][mask_index]["prompt"]
    aspect_ratio = data["crop"]["orientation"]
    image_hash = data["image_hash"]
    
    # Submit to Veo3
    result = veo3.submit_i2v(
        image_path=str(crop_path),
        prompt=mask_prompt,
        aspect_ratio=aspect_ratio
    )
    
    # Create veo3 directory
    veo3_dir = IMAGE_DIR / "annotations" / "veo3"
    veo3_dir.mkdir(exist_ok=True)
    
    # Use mask-based JSON naming
    mask_json_filename = f"{img_path.stem}-{image_hash}-mask-{mask_index + 1:03d}.json"
    mask_json_path = veo3_dir / mask_json_filename
    
    # Load or create mask JSON
    if mask_json_path.exists():
        with open(mask_json_path) as f:
            mask_data = json.load(f)
    else:
        mask_data = {
            "image_path": request['image_path'],
            "mask_index": mask_index,
            "jobs": []
        }
    
    # Add new job - Veo3 needs operation_data (base64 encoded for JSON)
    job = {
        "job_id": result['job_id'],
        "model": "veo3",
        "prompt": mask_prompt,
        "crop_filename": crop_filename,
        "aspect_ratio": aspect_ratio,
        "duration": 8,  # Veo3 always generates 8 seconds for i2v
        "status": result['status'],
        "operation_data": base64.b64encode(result['operation_data']).decode('utf-8'),  # Store as base64
        "created_at": datetime.now().isoformat()
    }
    
    mask_data["jobs"].append(job)
    
    with open(mask_json_path, "w") as f:
        json.dump(mask_data, f, indent=2)
    
    # Start polling for all incomplete jobs
    start_polling_all_jobs()
    
    return {
        "success": True,
        "job_id": result['job_id'],
        "status": result['status']
    }


def get_video_filename(image_path, image_hash, mask_index, job_index):
    """Derive video filename from job position"""
    img_path = Path(image_path)
    return f"{img_path.stem}-{image_hash}-mask-{mask_index + 1:03d}_video-{job_index + 1:03d}.mp4"


def poll_single_job(mask_json_path, job_index):
    """Poll a single generation job until complete"""
    import time
    
    # Determine model from path
    if 'sora2' in str(mask_json_path):
        from services import sora2 as service
        model_dir_name = 'sora2'
    elif 'veo3' in str(mask_json_path):
        from services import veo3 as service
        model_dir_name = 'veo3'
    else:
        print(f"Unknown model directory for {mask_json_path}")
        return
    
    try:
        with open(mask_json_path) as f:
            mask_data = json.load(f)
        
        if job_index >= len(mask_data['jobs']):
            return
        
        job = mask_data['jobs'][job_index]
        job_id = job['job_id']
        model = job['model']
        
        # Get annotation data to get image_hash
        img_path = IMAGE_DIR / mask_data['image_path']
        annotation_path = get_annotation_path(img_path)
        with open(annotation_path) as f:
            anno_data = json.load(f)
        image_hash = anno_data['image_hash']
        
        # Derive video filename
        video_filename = get_video_filename(
            mask_data['image_path'], 
            image_hash, 
            mask_data['mask_index'], 
            job_index
        )
        
        model_dir = IMAGE_DIR / "annotations" / model_dir_name
        video_path = model_dir / video_filename
        current_status = job.get('status')
        
        # Check if video already exists
        if video_path.exists():
            if current_status != 'completed':
                mask_data['jobs'][job_index]['status'] = 'completed'
                with open(mask_json_path, "w") as f:
                    json.dump(mask_data, f, indent=2)
                print(f"Fixed status for {job_id}: video exists but status was {current_status}")
            return
        
        # Prepare the identifier based on model
        if model == 'sora2':
            job_identifier = job_id
        elif model == 'veo3':
            # Decode the operation_data from base64
            job_identifier = base64.b64decode(job['operation_data'])
        else:
            print(f"Unknown model: {model}")
            return
        
        # If status is downloading, try to download immediately
        if current_status == 'downloading':
            try:
                service.download_video(job_identifier, str(video_path))
                mask_data['jobs'][job_index]['status'] = 'completed'
                with open(mask_json_path, "w") as f:
                    json.dump(mask_data, f, indent=2)
                print(f"Video re-downloaded: {video_filename}")
                return
            except Exception as e:
                print(f"Failed to re-download {job_id}, will poll for status: {e}")
        
        while True:
            time.sleep(30)
            
            try:
                status_result = service.check_status(job_identifier)
                
                # For Veo3, update the operation_data after each check
                if model == 'veo3' and 'operation_data' in status_result:
                    job_identifier = status_result['operation_data']
                    mask_data['jobs'][job_index]['operation_data'] = base64.b64encode(job_identifier).decode('utf-8')
                
                if status_result['status'] == 'completed':
                    # Set to downloading state
                    with open(mask_json_path) as f:
                        mask_data = json.load(f)
                    mask_data['jobs'][job_index]['status'] = 'downloading'
                    with open(mask_json_path, "w") as f:
                        json.dump(mask_data, f, indent=2)
                    
                    # Download video
                    service.download_video(job_identifier, str(video_path))
                    
                    # Mark as completed only after successful download
                    mask_data['jobs'][job_index]['status'] = 'completed'
                    with open(mask_json_path, "w") as f:
                        json.dump(mask_data, f, indent=2)
                    
                    print(f"Video downloaded: {video_filename}")
                    break
                    
                elif status_result['status'] == 'failed':
                    with open(mask_json_path) as f:
                        mask_data = json.load(f)
                    mask_data['jobs'][job_index]['status'] = 'failed'
                    with open(mask_json_path, "w") as f:
                        json.dump(mask_data, f, indent=2)
                    
                    print(f"Generation failed: {job_id}")
                    break
                
                else:
                    # Update status for in_progress/queued
                    with open(mask_json_path) as f:
                        mask_data = json.load(f)
                    mask_data['jobs'][job_index]['status'] = status_result['status']
                    with open(mask_json_path, "w") as f:
                        json.dump(mask_data, f, indent=2)
                    
            except Exception as e:
                print(f"Error polling {job_id}: {e}")
                break
                
    except Exception as e:
        print(f"Error in poll_single_job: {e}")


def start_polling_all_jobs():
    """Scan all mask JSONs and start polling for incomplete jobs"""
    import threading
    
    if not IMAGE_DIR.exists():
        return
    
    # Poll both sora2 and veo3 directories
    for model_name in ['sora2', 'veo3']:
        model_dir = IMAGE_DIR / "annotations" / model_name
        if not model_dir.exists():
            continue
        
        for json_file in model_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    mask_data = json.load(f)
                
                # Get annotation data to derive video filenames
                img_path = IMAGE_DIR / mask_data['image_path']
                annotation_path = get_annotation_path(img_path)
                with open(annotation_path) as f:
                    anno_data = json.load(f)
                image_hash = anno_data['image_hash']
                
                # Check each job in this mask
                for job_index, job in enumerate(mask_data.get('jobs', [])):
                    status = job.get('status')
                    job_id = job.get('job_id')
                    
                    # Derive video filename
                    video_filename = get_video_filename(
                        mask_data['image_path'],
                        image_hash,
                        mask_data['mask_index'],
                        job_index
                    )
                    video_path = model_dir / video_filename
                    video_exists = video_path.exists()
                    
                    # Fix inconsistent states
                    if status == 'completed' and not video_exists:
                        mask_data['jobs'][job_index]['status'] = 'downloading'
                        with open(json_file, "w") as f:
                            json.dump(mask_data, f, indent=2)
                        print(f"Fixed inconsistent state for {job_id}: marked as completed but video missing")
                        
                        thread = threading.Thread(
                            target=poll_single_job,
                            args=(json_file, job_index),
                            daemon=True
                        )
                        thread.start()
                        print(f"Started re-download for {job_id}")
                        
                    elif status in ['downloading', 'in_progress', 'queued', 'submitted'] and not video_exists:
                        thread = threading.Thread(
                            target=poll_single_job,
                            args=(json_file, job_index),
                            daemon=True
                        )
                        thread.start()
                        print(f"Started polling for {job_id}")
                        
            except Exception as e:
                print(f"Error checking job {json_file}: {e}")



@app.get("/api/generations")
def list_generations():
    """List all generation jobs across all images"""
    if not IMAGE_DIR.exists():
        return []
    
    jobs = []
    
    # Scan both sora2 and veo3 directories
    for model_name in ['sora2', 'veo3']:
        model_dir = IMAGE_DIR / "annotations" / model_name
        if not model_dir.exists():
            continue
        
        for json_file in sorted(model_dir.glob("*.json")):
            try:
                with open(json_file) as f:
                    mask_data = json.load(f)
                
                # Get annotation data to derive video filenames
                img_path = IMAGE_DIR / mask_data['image_path']
                annotation_path = get_annotation_path(img_path)
                with open(annotation_path) as f:
                    anno_data = json.load(f)
                image_hash = anno_data['image_hash']
                
                # Process each job in this mask
                for job_index, job in enumerate(mask_data.get('jobs', [])):
                    video_filename = get_video_filename(
                        mask_data['image_path'],
                        image_hash,
                        mask_data['mask_index'],
                        job_index
                    )
                    
                    video_path = model_dir / video_filename
                    video_exists = video_path.exists()
                    
                    jobs.append({
                        'job_id': job.get('job_id'),
                        'model': job.get('model'),
                        'image_path': mask_data.get('image_path'),
                        'mask_index': mask_data.get('mask_index'),
                        'prompt': job.get('prompt', ''),
                        'status': job.get('status'),
                        'video_filename': video_filename,
                        'video_exists': video_exists,
                        'created_at': job.get('created_at')
                    })
            except Exception as e:
                print(f"Error reading job {json_file}: {e}")
    
    # Sort by created_at, newest first
    jobs.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    
    return jobs


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
    """Start polling all incomplete jobs on server start"""
    if IMAGE_DIR and IMAGE_DIR.exists():
        start_polling_all_jobs()
    else:
        print(f"Image directory not found: {IMAGE_DIR}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, reload=True, log_level="debug")