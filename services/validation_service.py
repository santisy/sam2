import json
from pathlib import Path

# Global stats tracker
# Structure: {model: {"total": int, "annotated": int, "stats": {"localization": {True: int, False: int}, ...}}}
VALIDATION_STATS = {}

def _init_stats():
    """Initialize empty stats structure"""
    return {
        "total": 0,
        "annotated": 0,
        "stats": {
            "localization": {True: 0, False: 0},
            "articulation_type": {True: 0, False: 0},
            "sequence_plausible": {True: 0, False: 0},
        }
    }

def scan_all_validations(image_dir):
    """Scan all validation directories and build global stats on startup"""
    global VALIDATION_STATS
    VALIDATION_STATS = {}
    
    models = ["ours", "rgb", "wan", "sora2", "veo3", "kling"]
    
    for model in models:
        VALIDATION_STATS[model] = _init_stats()
        
        video_dir = image_dir / "annotations" / model
        validation_dir = video_dir / "validation"
        
        if not video_dir.exists():
            continue
        
        # Count total videos (video-001 files only)
        video_files = list(video_dir.glob("*_video-001.mp4"))
        VALIDATION_STATS[model]["total"] = len(video_files)
        
        # Load validation jsons if they exist
        if validation_dir.exists():
            for video_file in video_files:
                validation_file = validation_dir / video_file.name.replace(".mp4", ".json")
                if validation_file.exists():
                    with open(validation_file) as f:
                        data = json.load(f)
                        
                        # Check if any field is annotated
                        has_annotation = any(
                            data.get(field) is not None 
                            for field in ["localization", "articulation_type", "sequence_plausible"]
                        )
                        
                        if has_annotation:
                            VALIDATION_STATS[model]["annotated"] += 1
                            
                            # Update field stats
                            for field in ["localization", "articulation_type", "sequence_plausible"]:
                                value = data.get(field)
                                if value is not None:
                                    VALIDATION_STATS[model]["stats"][field][value] += 1

def get_global_stats():
    """Return current global stats"""
    return VALIDATION_STATS

def get_image_stats(image_dir, img_path):
    """Get validation stats for a specific image - average completed models per mask for each metric"""
    annotation_path = image_dir / "annotations" / f"{img_path.stem}.json"
    if not annotation_path.exists():
        return {"localization": 0, "articulation_type": 0, "sequence_plausible": 0, "perspective": 0}
    
    with open(annotation_path) as f:
        data = json.load(f)
        masks = data.get("masks", [])
        
    if not masks:
        return {"localization": 0, "articulation_type": 0, "sequence_plausible": 0, "perspective": 0}
    
    models = ["ours", "rgb", "wan", "sora2", "veo3", "kling"]
    metrics = ["localization", "articulation_type", "sequence_plausible", "perspective"]
    
    # Calculate average completed models per mask for each metric
    metric_totals = {metric: 0 for metric in metrics}
    
    for mask_index in range(len(masks)):
        for model in models:
            validation_path = get_validation_path(image_dir, img_path, mask_index, model)
            if validation_path and validation_path.exists():
                with open(validation_path) as f:
                    validation_data = json.load(f)
                    for metric in metrics:
                        if validation_data.get(metric) is not None:
                            metric_totals[metric] += 1
    
    # Calculate averages
    num_masks = len(masks)
    averages = {metric: round(metric_totals[metric] / num_masks, 1) if num_masks > 0 else 0 for metric in metrics}
    
    return averages

def _update_global_stats(model, old_data, new_data):
    """Update global stats when annotation changes"""
    if model not in VALIDATION_STATS:
        return
    
    # Check if this is newly annotated
    old_annotated = any(old_data.get(field) is not None for field in ["localization", "articulation_type", "sequence_plausible"])
    new_annotated = any(new_data.get(field) is not None for field in ["localization", "articulation_type", "sequence_plausible"])
    
    if not old_annotated and new_annotated:
        VALIDATION_STATS[model]["annotated"] += 1
    elif old_annotated and not new_annotated:
        VALIDATION_STATS[model]["annotated"] -= 1
    
    # Update field stats
    for field in ["localization", "articulation_type", "sequence_plausible"]:
        old_value = old_data.get(field)
        new_value = new_data.get(field)
        
        if old_value != new_value:
            # Decrement old value
            if old_value is not None:
                VALIDATION_STATS[model]["stats"][field][old_value] -= 1
            
            # Increment new value
            if new_value is not None:
                VALIDATION_STATS[model]["stats"][field][new_value] += 1

def get_validation_path(image_dir, img_path, mask_index, model):
    """Get path to validation json file for image/mask/model combination"""
    annotations_dir = image_dir / "annotations" / model / "validation"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Get mask filename from annotation to construct validation filename
    annotation_path = image_dir / "annotations" / f"{img_path.stem}.json"
    if not annotation_path.exists():
        return None
    
    with open(annotation_path) as f:
        data = json.load(f)
        if mask_index >= len(data.get("masks", [])):
            return None
        mask_filename = data["masks"][mask_index]["mask_filename"]
        # Convert mask filename to validation filename
        # e.g., "1f694ae2-9b79ec-mask-001.png" -> "1f694ae2-9b79ec-mask-001_video-001.json"
        validation_filename = mask_filename.replace(".png", "_video-001.json")
        return annotations_dir / validation_filename

def get_validation(image_dir, img_path, mask_index, model):
    """Get validation data for image/mask/model combination"""
    validation_path = get_validation_path(image_dir, img_path, mask_index, model)
    if not validation_path or not validation_path.exists():
        return {
            "localization": None,
            "articulation_type": None,
            "sequence_plausible": None,
            "perspective": None
        }
    
    with open(validation_path) as f:
        return json.load(f)

def update_validation(image_dir, img_path, mask_index, model, field, value):
    """Update a single validation field"""
    validation_path = get_validation_path(image_dir, img_path, mask_index, model)
    if not validation_path:
        return {"success": False, "error": "Invalid mask index"}
    
    # Load existing or create new
    if validation_path.exists():
        with open(validation_path) as f:
            old_data = json.load(f)
    else:
        old_data = {
            "localization": None,
            "articulation_type": None,
            "sequence_plausible": None,
            "perspective": None
        }
    
    # Update field
    new_data = old_data.copy()
    new_data[field] = value
    
    # Save
    with open(validation_path, 'w') as f:
        json.dump(new_data, f, indent=2)
    
    # Update global stats
    _update_global_stats(model, old_data, new_data)
    
    return {"success": True, "data": new_data}