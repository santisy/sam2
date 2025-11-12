import os
import fal_client
import requests
from services.api_keys import FAL_API_KEY

# fal_client expects FAL_KEY environment variable
os.environ['FAL_KEY'] = FAL_API_KEY

MODEL = "fal-ai/kling-video/v2.5-turbo/standard/image-to-video"


def submit_i2v(image_path, prompt, aspect_ratio):
    """
    Submit image-to-video generation job to Kling 2.5 Turbo Standard
    
    aspect_ratio: 'landscape' or 'portrait'
    returns: {'job_id': '...', 'status': '...'}
    """
    # Upload image file - returns URL
    image_url = fal_client.upload_file(image_path)
    
    # Map aspect ratio
    if aspect_ratio == "landscape":
        ar = "16:9"
    else:
        ar = "9:16"
    
    # Submit to queue (returns SyncRequestHandle object)
    response = fal_client.submit(
        MODEL,
        arguments={
            "prompt": prompt,
            "image_url": image_url,
            "aspect_ratio": ar,
            "duration": "5"
        }
    )
    
    return {
        'job_id': response.request_id,
        'status': 'queued'
    }


def check_status(job_id):
    """
    Check status of generation job
    
    returns: {'status': 'queued|in_progress|completed|failed', 'progress': 0-100}
    """
    status_obj = fal_client.status(MODEL, request_id=job_id, with_logs=False)
    
    if isinstance(status_obj, fal_client.Queued):
        return {
            'status': 'queued',
            'progress': 0
        }
    elif isinstance(status_obj, fal_client.InProgress):
        return {
            'status': 'in_progress',
            'progress': 0
        }
    elif isinstance(status_obj, fal_client.Completed):
        return {
            'status': 'completed',
            'progress': 100
        }
    else:
        status_str = str(type(status_obj).__name__).lower()
        if 'fail' in status_str:
            return {'status': 'failed', 'progress': 0}
        return {
            'status': 'unknown',
            'progress': 0
        }


def download_video(job_id, output_path):
    """
    Download completed video
    """
    # Get result from queue - returns actual result data
    result = fal_client.result(MODEL, request_id=job_id)
    
    # Result is the actual data dict: {'video': {'url': '...'}}
    # Handle both dict and object attribute access
    if isinstance(result, dict):
        video_url = result['video']['url']
    else:
        video_url = result.video.url
    
    # Download video
    video_data = requests.get(video_url).content
    
    with open(output_path, 'wb') as f:
        f.write(video_data)