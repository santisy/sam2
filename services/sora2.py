from openai import OpenAI
from PIL import Image
import io
from services.api_keys import SORA2_API_KEY

client = OpenAI(api_key=SORA2_API_KEY)

def submit_i2v(image_path, prompt, aspect_ratio):
    """
    Submit image-to-video generation job to Sora2
    
    aspect_ratio: 'landscape' or 'portrait'
    returns: {'job_id': '...', 'status': '...'}
    """
    if aspect_ratio == "landscape":
        size = "1280x720"
        target_width, target_height = 1280, 720
    else:
        size = "720x1280"
        target_width, target_height = 720, 1280
    
    # Resize image to exactly match requested dimensions
    img = Image.open(image_path)
    img_resized = img.resize((target_width, target_height), Image.LANCZOS)
    
    # Convert to bytes with proper metadata
    img_bytes = io.BytesIO()
    img_resized.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    img_bytes.name = 'image.png'  # Add filename for proper mimetype detection
    
    response = client.videos.create(
        model="sora-2",
        prompt=prompt,
        size=size,
        seconds="4",
        input_reference=img_bytes
    )
    
    return {
        'job_id': response.id,
        'status': response.status
    }

def check_status(job_id):
    """
    Check status of generation job
    
    returns: {'status': 'queued|in_progress|completed|failed', 'progress': 0-100}
    """
    response = client.videos.retrieve(job_id)
    
    return {
        'status': response.status,
        'progress': getattr(response, 'progress', 0)
    }

def download_video(job_id, output_path):
    """
    Download completed video
    """
    content = client.videos.download_content(job_id)
    
    with open(output_path, 'wb') as f:
        f.write(content.read())