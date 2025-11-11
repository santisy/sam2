import pickle
import os
from google import genai
from google.genai.types import GenerateVideosConfig, Image
from PIL import Image as PILImage
from services.api_keys import VERTEX_API_KEY
import time

# Setup
os.environ['GOOGLE_API_KEY'] = VERTEX_API_KEY
os.environ['GOOGLE_CLOUD_PROJECT'] = 'gen-lang-client-0798041980'
os.environ['GOOGLE_CLOUD_LOCATION'] = 'us-central1'
os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'True'

client = genai.Client()

def submit_i2v(image_path, prompt, aspect_ratio, output_gcs_uri=None):
    """
    Submit image-to-video generation job to Veo 3.1 Fast
    
    aspect_ratio: 'landscape' or 'portrait'
    returns: {'job_id': '...', 'status': '...'}
    """
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
            resolution="720p",
            number_of_videos=1
        )
    )
    
    # Pickle the operation object to store it
    job_id = operation.name  # Use name as unique ID
    operation_data = pickle.dumps(operation)
    
    return {
        'job_id': job_id,
        'operation_data': operation_data,  # Store this in your DB/session
        'status': 'submitted'
    }

def check_status(operation_data):
    """
    Check status of generation job
    
    operation_data: pickled operation object from submit_i2v
    returns: {'status': '...', 'done': bool, 'operation_data': ...}
    """
    # Unpickle the operation
    operation = pickle.loads(operation_data)
    
    # Refresh the operation status
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
    """
    Download completed video
    
    operation_data: pickled operation object
    """
    operation = pickle.loads(operation_data)
    
    if not operation.done:
        raise Exception("Operation not complete yet")
    
    if operation.error:
        raise Exception(f"Operation failed: {operation.error}")
    
    # Get the result
    result = operation.response
    
    # Download the video
    if hasattr(result, 'generated_videos') and result.generated_videos:
        video = result.generated_videos[0]
        client.files.download(file=video.video)
        video.video.save(output_path)
    else:
        raise Exception("No video found in result")

if __name__ == "__main__":
    check_status("projects/gen-lang-client-0798041980/locations/us-central1/publishers/google/models/veo-3.1-fast-generate-preview/operations/78293bd3-9bbe-44d2-8f9b-74c09e49ea0f")