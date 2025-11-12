import json
import base64
import threading
import time
from pathlib import Path
from datetime import datetime


# Central job registry: {job_id: {'status': ..., 'video_path': ..., ...}}
ACTIVE_JOBS = {}
JOBS_LOCK = threading.Lock()


def get_video_filename(image_path, image_hash, mask_index, job_index):
    img_path = Path(image_path)
    return f"{img_path.stem}-{image_hash}-mask-{mask_index + 1:03d}_video-{job_index + 1:03d}.mp4"


def poll_job_thread(job_id, model, job_identifier, video_path):
    if model == 'sora2':
        from services import sora2 as service
    elif model == 'veo3':
        from services import veo3 as service
    else:
        return
    
    while True:
        time.sleep(30)
        
        try:
            if video_path.exists():
                with JOBS_LOCK:
                    if job_id in ACTIVE_JOBS:
                        ACTIVE_JOBS[job_id]['status'] = 'completed'
                print(f"Video exists: {video_path.name}")
                break
            
            status_result = service.check_status(job_identifier)
            
            with JOBS_LOCK:
                if job_id in ACTIVE_JOBS:
                    ACTIVE_JOBS[job_id]['status'] = status_result['status']
                    if model == 'veo3' and 'operation_data' in status_result:
                        ACTIVE_JOBS[job_id]['operation_data'] = status_result['operation_data']
                        job_identifier = status_result['operation_data']
            
            if status_result['status'] == 'completed':
                service.download_video(job_identifier, str(video_path))
                with JOBS_LOCK:
                    if job_id in ACTIVE_JOBS:
                        ACTIVE_JOBS[job_id]['status'] = 'completed'
                print(f"Downloaded: {video_path.name}")
                break
            elif status_result['status'] == 'failed':
                print(f"Failed: {job_id}")
                break
                
        except Exception as e:
            print(f"Error polling {job_id}: {e}")
            break


def load_all_jobs(image_dir):
    global ACTIVE_JOBS
    
    if not image_dir.exists():
        return
    
    with JOBS_LOCK:
        ACTIVE_JOBS.clear()
    
    for model_name in ['sora2', 'veo3']:
        model_dir = image_dir / "annotations" / model_name
        if not model_dir.exists():
            continue
        
        for json_file in model_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    mask_data = json.load(f)
                
                from services import mask_crop_service as mcs
                img_path = image_dir / mask_data['image_path']
                annotation_path = mcs.get_annotation_path(image_dir, img_path)
                with open(annotation_path) as f:
                    anno_data = json.load(f)
                image_hash = anno_data['image_hash']
                
                for job_index, job in enumerate(mask_data.get('jobs', [])):
                    video_filename = get_video_filename(
                        mask_data['image_path'],
                        image_hash,
                        mask_data['mask_index'],
                        job_index
                    )
                    video_path = model_dir / video_filename
                    
                    job_info = {
                        'job_id': job['job_id'],
                        'model': job['model'],
                        'image_path': mask_data['image_path'],
                        'mask_index': mask_data['mask_index'],
                        'prompt': job.get('prompt', ''),
                        'video_filename': video_filename,
                        'video_path': video_path,
                        'created_at': job.get('created_at'),
                        'status': 'completed' if video_path.exists() else 'in_progress'
                    }
                    
                    with JOBS_LOCK:
                        ACTIVE_JOBS[job['job_id']] = job_info
                    
                    if not video_path.exists():
                        if job['model'] == 'sora2':
                            job_identifier = job['job_id']
                        elif job['model'] == 'veo3':
                            job_identifier = base64.b64decode(job['operation_data'])
                        else:
                            continue
                        
                        thread = threading.Thread(
                            target=poll_job_thread,
                            args=(job['job_id'], job['model'], job_identifier, video_path),
                            daemon=True
                        )
                        thread.start()
                        print(f"Started polling: {job['job_id']}")
                        
            except Exception as e:
                print(f"Error loading job {json_file}: {e}")


def async_submit_job(temp_job_id, image_dir, image_path, mask_index, image_hash, crop_path, mask_prompt, aspect_ratio, model):
    if model == 'sora2':
        from services import sora2 as service
        model_dir_name = 'sora2'
    elif model == 'veo3':
        from services import veo3 as service
        model_dir_name = 'veo3'
    else:
        return
    
    try:
        img_path = Path(image_path)
        
        result = service.submit_i2v(
            image_path=str(crop_path),
            prompt=mask_prompt,
            aspect_ratio=aspect_ratio
        )
        
        model_dir = image_dir / "annotations" / model_dir_name
        model_dir.mkdir(exist_ok=True)
        
        mask_json_filename = f"{img_path.stem}-{image_hash}-mask-{mask_index + 1:03d}.json"
        mask_json_path = model_dir / mask_json_filename
        
        if mask_json_path.exists():
            with open(mask_json_path) as f:
                mask_data = json.load(f)
        else:
            mask_data = {
                "image_path": image_path,
                "mask_index": mask_index,
                "jobs": []
            }
        
        job_index = len(mask_data["jobs"])
        
        job = {
            "job_id": result['job_id'],
            "model": model,
            "prompt": mask_prompt,
            "crop_filename": crop_path.name,
            "aspect_ratio": aspect_ratio,
            "duration": 4,
            "created_at": datetime.now().isoformat()
        }
        
        if model == 'veo3':
            job['operation_data'] = base64.b64encode(result['operation_data']).decode('utf-8')
        
        mask_data["jobs"].append(job)
        
        with open(mask_json_path, "w") as f:
            json.dump(mask_data, f, indent=2)
        
        video_filename = get_video_filename(image_path, image_hash, mask_index, job_index)
        video_path = model_dir / video_filename
        
        real_job_id = result['job_id']
        
        with JOBS_LOCK:
            # Remove temp entry, add real one
            if temp_job_id in ACTIVE_JOBS:
                temp_info = ACTIVE_JOBS.pop(temp_job_id)
            
            ACTIVE_JOBS[real_job_id] = {
                'job_id': real_job_id,
                'model': model,
                'image_path': image_path,
                'mask_index': mask_index,
                'prompt': mask_prompt,
                'video_filename': video_filename,
                'video_path': video_path,
                'created_at': job['created_at'],
                'status': 'queued'
            }
        
        if model == 'sora2':
            job_identifier = result['job_id']
        elif model == 'veo3':
            job_identifier = result['operation_data']
        
        thread = threading.Thread(
            target=poll_job_thread,
            args=(real_job_id, model, job_identifier, video_path),
            daemon=True
        )
        thread.start()
        
    except Exception as e:
        print(f"Error in async_submit_job: {e}")
        with JOBS_LOCK:
            if temp_job_id in ACTIVE_JOBS:
                ACTIVE_JOBS[temp_job_id]['status'] = 'failed'


def submit_job(image_dir, image_path, mask_index, image_hash, crop_path, mask_prompt, aspect_ratio, model):
    import uuid
    
    temp_job_id = f"pending_{uuid.uuid4().hex[:8]}"
    
    model_dir_name = 'sora2' if model == 'sora2' else 'veo3'
    model_dir = image_dir / "annotations" / model_dir_name
    
    video_filename = get_video_filename(image_path, image_hash, mask_index, 999)  # Temp
    video_path = model_dir / video_filename
    
    with JOBS_LOCK:
        ACTIVE_JOBS[temp_job_id] = {
            'job_id': temp_job_id,
            'model': model,
            'image_path': image_path,
            'mask_index': mask_index,
            'prompt': mask_prompt,
            'video_filename': video_filename,
            'video_path': video_path,
            'created_at': datetime.now().isoformat(),
            'status': 'submitting'
        }
    
    thread = threading.Thread(
        target=async_submit_job,
        args=(temp_job_id, image_dir, image_path, mask_index, image_hash, crop_path, mask_prompt, aspect_ratio, model),
        daemon=True
    )
    thread.start()
    
    return temp_job_id


def list_all_jobs():
    with JOBS_LOCK:
        jobs = []
        for job_id, info in ACTIVE_JOBS.items():
            jobs.append({
                'job_id': job_id,
                'model': info['model'],
                'image_path': info['image_path'],
                'mask_index': info['mask_index'],
                'prompt': info['prompt'],
                'status': info['status'],
                'video_filename': info['video_filename'],
                'video_exists': info['video_path'].exists(),
                'created_at': info['created_at']
            })
        
        jobs.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return jobs