import os
import json
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image


def get_image_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:6]


def get_annotation_path(image_dir, image_path):
    annotations_dir = image_dir / "annotations"
    annotations_dir.mkdir(exist_ok=True)
    return annotations_dir / f"{image_path.stem}.json"


def load_annotation_data(annotation_path):
    if not annotation_path.exists():
        return None
    with open(annotation_path) as f:
        return json.load(f)


def save_annotation_data(annotation_path, data):
    with open(annotation_path, "w") as f:
        json.dump(data, f, indent=2)


def create_mask(image_dir, image_path, mask_array, prompt):
    annotations_dir = image_dir / "annotations"
    annotations_dir.mkdir(exist_ok=True)
    
    image_hash = get_image_hash(image_path)
    annotation_path = get_annotation_path(image_dir, image_path)
    
    data = load_annotation_data(annotation_path)
    if data is None:
        data = {
            "image_filename": image_path.name,
            "image_hash": image_hash,
            "masks": []
        }
    
    mask_img = Image.fromarray((mask_array.astype(np.uint8)) * 255)
    
    mask_index = len(data["masks"]) + 1
    mask_filename = f"{image_path.stem}-{image_hash}-mask-{mask_index:03d}.png"
    mask_path = annotations_dir / mask_filename
    mask_img.save(mask_path)
    
    data["masks"].append({
        "mask_filename": mask_filename,
        "prompt": prompt
    })
    
    save_annotation_data(annotation_path, data)
    
    return mask_index - 1


def update_mask(image_dir, image_path, mask_index, mask_array, prompt):
    annotation_path = get_annotation_path(image_dir, image_path)
    data = load_annotation_data(annotation_path)
    
    if data is None or mask_index >= len(data["masks"]):
        return False
    
    annotations_dir = image_dir / "annotations"
    mask_filename = data["masks"][mask_index]["mask_filename"]
    mask_path = annotations_dir / mask_filename
    
    mask_img = Image.fromarray((mask_array.astype(np.uint8)) * 255)
    mask_img.save(mask_path)
    
    data["masks"][mask_index]["prompt"] = prompt
    
    save_annotation_data(annotation_path, data)
    
    return True


def delete_mask(image_dir, image_path, mask_index):
    annotation_path = get_annotation_path(image_dir, image_path)
    data = load_annotation_data(annotation_path)
    
    if data is None or mask_index >= len(data["masks"]):
        return False
    
    annotations_dir = image_dir / "annotations"
    mask_filename = data["masks"][mask_index]["mask_filename"]
    mask_path = annotations_dir / mask_filename
    
    if mask_path.exists():
        os.remove(mask_path)
    
    data["masks"].pop(mask_index)
    
    save_annotation_data(annotation_path, data)
    
    return True


def get_annotation_with_masks(image_dir, image_path):
    annotation_path = get_annotation_path(image_dir, image_path)
    data = load_annotation_data(annotation_path)
    
    if data is None:
        return {"masks": []}
    
    annotations_dir = image_dir / "annotations"
    for mask_info in data.get("masks", []):
        mask_path = annotations_dir / mask_info["mask_filename"]
        if mask_path.exists():
            mask_img = Image.open(mask_path)
            mask_array = np.array(mask_img) > 127
            mask_info["mask_data"] = mask_array.tolist()
    
    return data


def save_crop(image_dir, image_path, crop_info):
    annotation_path = get_annotation_path(image_dir, image_path)
    image_hash = get_image_hash(image_path)
    
    data = load_annotation_data(annotation_path)
    if data:
        if "crop" in data and "crop_filename" in data["crop"]:
            crops_dir = image_dir / "annotations" / "crops"
            old_crop_path = crops_dir / data["crop"]["crop_filename"]
            if old_crop_path.exists():
                os.remove(old_crop_path)
    else:
        annotations_dir = image_dir / "annotations"
        annotations_dir.mkdir(exist_ok=True)
        data = {
            "image_filename": image_path.name,
            "image_hash": image_hash,
            "masks": []
        }
    
    crops_dir = image_dir / "annotations" / "crops"
    crops_dir.mkdir(exist_ok=True)
    
    image = Image.open(image_path)
    x = int(crop_info['x'])
    y = int(crop_info['y'])
    width = int(crop_info['width'])
    height = int(crop_info['height'])
    
    cropped = image.crop((x, y, x + width, y + height))
    
    crop_filename = f"{image_path.stem}-{image_hash}-crop.png"
    crop_path = crops_dir / crop_filename
    cropped.save(crop_path, "PNG")
    
    crop_info['crop_filename'] = crop_filename
    data["crop"] = crop_info
    
    save_annotation_data(annotation_path, data)
    
    return True


def delete_crop(image_dir, image_path):
    annotation_path = get_annotation_path(image_dir, image_path)
    data = load_annotation_data(annotation_path)
    
    if data and "crop" in data:
        if "crop_filename" in data["crop"]:
            crops_dir = image_dir / "annotations" / "crops"
            crop_path = crops_dir / data["crop"]["crop_filename"]
            if crop_path.exists():
                os.remove(crop_path)
        
        del data["crop"]
        save_annotation_data(annotation_path, data)
    
    return True