from google import genai
from google.genai.types import GenerateVideosConfig, Image
from PIL import Image as PILImage
from services.api_keys import VERTEX_API_KEY
import os
import time
import pickle

# Setup
os.environ['GOOGLE_API_KEY'] = VERTEX_API_KEY
os.environ['GOOGLE_CLOUD_PROJECT'] = 'gen-lang-client-0798041980'
os.environ['GOOGLE_CLOUD_LOCATION'] = 'us-central1'
os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'True'

client = genai.Client()


def submit_i2v(image_path, prompt, aspect_ratio, output_gcs_uri=None):
    if aspect_ratio == "landscape":
        aspect_ratio_str = "16:9"
        target_width, target_height = 1280, 720
    else:
        aspect_ratio_str = "9:16"
        target_width, target_height = 720, 1280
    
    # Resize image
    img = PILImage.open(image_path)
    img_resized = img.resize((target_width, target_height), PILImage.LANCZOS)
    
    # Save to bytes
    import io
    img_bytes = io.BytesIO()
    img_resized.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Submit operation
    # Note: Veo 3.1 Fast i2v always generates 8-second videos
    operation = client.models.generate_videos(
        model="veo-3.1-fast-generate-preview",
        prompt=prompt,
        image=Image(
            image_bytes=img_bytes.read(),
            mime_type="image/png"
        ),
        config=GenerateVideosConfig(
            aspect_ratio=aspect_ratio_str,
            output_gcs_uri=output_gcs_uri,
            resolution="720p",  # Can be "720p" or "1080p"
            number_of_videos=1,
            duration_seconds=4,
        )
    )
    
    # Pickle the operation object to store it
    job_id = operation.name  # Use name as unique ID
    operation_data = pickle.dumps(operation)
    
    return {
        'job_id': job_id,
        'operation_data': operation_data,  # Return pickled bytes
        'status': 'submitted'
    }


def check_status(operation_data):
    # Unpickle the operation
    operation = pickle.loads(operation_data)
    
    # Refresh the operation status from API
    operation = client.operations.get(operation)
    
    # Re-pickle for storage
    updated_operation_data = pickle.dumps(operation)
    
    if operation.done:
        if operation.error:
            return {
                'status': 'failed',
                'done': True,
                'error': str(operation.error),
                'operation_data': updated_operation_data
            }
        return {
            'status': 'completed',
            'done': True,
            'operation_data': updated_operation_data
        }
    
    return {
        'status': 'in_progress',
        'done': False,
        'progress': 0,
        'operation_data': updated_operation_data
    }


def download_video(operation_data, output_path):
    operation = pickle.loads(operation_data)
    
    if not operation.done:
        raise Exception("Operation not complete yet")
    
    if operation.error:
        raise Exception(f"Operation failed: {operation.error}")
    
    # Get the result
    result = operation.response
    
    # Write video bytes directly to file
    if hasattr(result, 'generated_videos') and result.generated_videos:
        video = result.generated_videos[0]
        
        # The video bytes are already in the response
        if hasattr(video.video, 'video_bytes') and video.video.video_bytes:
            with open(output_path, 'wb') as f:
                f.write(video.video.video_bytes)
        else:
            raise Exception("No video_bytes found in response")
    else:
        raise Exception("No video found in result")


def wait_for_completion(operation_data, poll_interval=15, timeout=600):
    start_time = time.time()
    
    while True:
        status = check_status(operation_data)
        operation_data = status['operation_data']  # Update with latest
        
        if status['done']:
            if status['status'] == 'failed':
                raise Exception(f"Generation failed: {status.get('error')}")
            return True, operation_data
        
        if time.time() - start_time > timeout:
            return False, operation_data
        
        print(f"Waiting... {int(time.time() - start_time)}s elapsed")
        time.sleep(poll_interval)