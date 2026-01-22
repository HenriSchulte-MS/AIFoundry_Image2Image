import os
import requests
import base64
from PIL import Image
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv
import argparse
import time

load_dotenv()

# Read environment variables
FOUNDRY_ENDPOINT = os.getenv("FOUNDRY_ENDPOINT")
FOUNDRY_API_KEY = os.getenv("FOUNDRY_API_KEY")
FOUNDRY_API_VERSION = os.getenv("FOUNDRY_API_VERSION", "2025-04-01-preview")
FLUX_DEPLOYMENT_NAME = os.getenv("FLUX_DEPLOYMENT_NAME")
GPT_DEPLOYMENT_NAME = os.getenv("GPT_DEPLOYMENT_NAME")
INPUT_IMAGE = os.getenv("INPUT_IMAGE")
INPUT_IMAGE_2 = os.getenv("INPUT_IMAGE_2")  # Optional second image for Flux-2
PROMPT = os.getenv("PROMPT")

# Maximum image size limits (in megapixels)
# FLUX supports up to 4MP, using 4MP for max quality
MAX_IMAGE_MP = float(os.getenv("MAX_IMAGE_MP", "4.0"))  # Default 4MP (FLUX maximum)
MAX_IMAGE_DIMENSION = int(os.getenv("MAX_IMAGE_DIMENSION", "2048"))  # Max dimension in pixels


def resize_image_if_needed(image_path: str, max_mp: float = MAX_IMAGE_MP, 
                           max_dimension: int = MAX_IMAGE_DIMENSION) -> tuple:
    """
    Resize image if it exceeds size limits, preserving aspect ratio.
    
    Args:
        image_path: Path to the image file
        max_mp: Maximum megapixels (default 4.0, FLUX max is 4.0)
        max_dimension: Maximum width or height in pixels
    
    Returns:
        Tuple of (image_bytes, width, height) - PNG format, resized if necessary
    """
    with Image.open(image_path) as img:
        original_width, original_height = img.size
        original_mp = (original_width * original_height) / 1_000_000
        
        new_width, new_height = original_width, original_height
        needs_resize = False
        
        # Check if image exceeds megapixel limit
        if original_mp > max_mp:
            scale = (max_mp / original_mp) ** 0.5
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            needs_resize = True
        
        # Check if any dimension exceeds max
        if new_width > max_dimension or new_height > max_dimension:
            scale = min(max_dimension / new_width, max_dimension / new_height)
            new_width = int(new_width * scale)
            new_height = int(new_height * scale)
            needs_resize = True
        
        # Ensure dimensions are multiples of 16 (required by FLUX)
        new_width = (new_width // 16) * 16
        new_height = (new_height // 16) * 16
        
        # Ensure minimum dimension of 64 pixels
        new_width = max(64, new_width)
        new_height = max(64, new_height)
        
        if needs_resize:
            print(f"  Resizing image from {original_width}x{original_height} ({original_mp:.2f}MP) "
                  f"to {new_width}x{new_height} ({(new_width * new_height) / 1_000_000:.2f}MP)")
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            # Even if not resizing, ensure dimensions are multiples of 16
            if original_width != new_width or original_height != new_height:
                print(f"  Adjusting dimensions to multiples of 16: {original_width}x{original_height} -> {new_width}x{new_height}")
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to RGB if necessary (e.g., RGBA -> RGB for JPEG compatibility)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        # Save to bytes
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue(), new_width, new_height


def image_to_base64(image_path: str, resize: bool = True) -> tuple:
    """
    Read an image file and return its base64-encoded string.
    
    Args:
        image_path: Path to the image file
        resize: Whether to resize large images (default True)
    
    Returns:
        Tuple of (base64_string, width, height)
    """
    if resize:
        image_bytes, width, height = resize_image_if_needed(image_path)
        return base64.b64encode(image_bytes).decode("utf-8"), width, height
    else:
        with open(image_path, "rb") as f:
            data = f.read()
        with Image.open(image_path) as img:
            width, height = img.size
        return base64.b64encode(data).decode("utf-8"), width, height


def get_resized_image_bytes(image_path: str) -> bytes:
    """Get resized image bytes for multipart upload."""
    image_bytes, _, _ = resize_image_if_needed(image_path)
    return image_bytes


def call_gpt_image_edit(client_endpoint: str, api_key: str, api_version: str, 
                        deployment: str, image_path: str, prompt: str) -> dict:
    """Call the GPT Image Edit API using multipart/form-data."""
    edit_url = f"{client_endpoint}openai/deployments/{deployment}/images/edits?api-version={api_version}"
    
    # GPT image edit supports specific sizes: 256x256, 512x512, 1024x1024, 1536x1536, 
    # 1792x1024, 1024x1792, or "auto"
    # Using "auto" to let the API determine best size based on input
    form_data = {
        "prompt": (None, prompt),
        "model": (None, deployment),
        "size": (None, "auto"),  # Let API determine size based on input
        "n": (None, "1"),
        "input_fidelity": (None, "high"),
        "quality": (None, "high"),
    }
    
    # Resize image if needed before upload
    print(f"  Processing input image: {image_path}")
    image_bytes = get_resized_image_bytes(image_path)
    
    files = {
        **form_data,
        "image": (os.path.basename(image_path), image_bytes, "image/png"),
    }
    response = requests.post(
        edit_url,
        headers={"api-key": api_key},
        files=files,
    )
    return response.json()


def call_flux_image_edit(client_endpoint: str, api_key: str, api_version: str,
                         deployment: str, image_path: str, prompt: str,
                         image_path_2: str = None,
                         image_path_3: str = None,
                         image_path_4: str = None,
                         image_path_5: str = None,
                         image_path_6: str = None,
                         image_path_7: str = None,
                         image_path_8: str = None,
                         output_format: str = "png") -> dict:
    """
    Call the Flux-2-pro API using JSON body with base64-encoded images.
    
    Supports up to 8 reference images for multi-reference editing.
    Based on: https://docs.bfl.ai/flux_2/flux2_image_editing
    """
    # Flux-2-pro uses Black Forest Labs provider endpoint
    edit_url = f"{client_endpoint}providers/blackforestlabs/v1/flux-2-pro?api-version={api_version}"
    
    # Process and resize main input image
    print(f"  Processing input image: {image_path}")
    input_b64, input_width, input_height = image_to_base64(image_path)
    print(f"  Output will be: {input_width}x{input_height}")
    
    request_body = {
        "model": deployment,
        "prompt": prompt,
        "output_format": output_format,
        "input_image": input_b64,
        "width": input_width,    # Explicitly set output dimensions
        "height": input_height,  # to match processed input
        "n": 1,
    }
    
    # Add optional reference images (up to 8 total via API)
    reference_images = [
        ("input_image_2", image_path_2),
        ("input_image_3", image_path_3),
        ("input_image_4", image_path_4),
        ("input_image_5", image_path_5),
        ("input_image_6", image_path_6),
        ("input_image_7", image_path_7),
        ("input_image_8", image_path_8),
    ]
    
    for key, path in reference_images:
        if path and os.path.exists(path):
            print(f"  Processing {key}: {path}")
            ref_b64, _, _ = image_to_base64(path)  # Will resize if needed
            request_body[key] = ref_b64
    
    response = requests.post(
        edit_url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json=request_body,
        timeout=300,  # 5 minute timeout for image generation
    )
    return response.json()


if __name__ == "__main__":

    # Parse model selection argument
    parser = argparse.ArgumentParser(description="Image to Image Processing with AIFoundry")
    parser.add_argument("-model", "--model", dest="model", type=str, 
                        help="Model to use: 'gpt' (gpt-image-1.5) or 'flux' (FLUX.2-pro)")
    args = parser.parse_args()

    if args.model:
        model = args.model.lower()
        if model == "gpt":
            deployment = GPT_DEPLOYMENT_NAME
        else:
            deployment = FLUX_DEPLOYMENT_NAME
            model = "flux"
        print(f"Using {deployment} model.")
    else:
        model = "flux"
        deployment = FLUX_DEPLOYMENT_NAME
        print(f"No -model argument provided. Using {deployment} model.")

    print(f"Sending edit request for image {INPUT_IMAGE} with prompt: {PROMPT} ...")

    # Time the API request
    start = time.perf_counter()

    # Call the appropriate API based on model selection
    if model == "gpt":
        response_json = call_gpt_image_edit(
            FOUNDRY_ENDPOINT, FOUNDRY_API_KEY, FOUNDRY_API_VERSION,
            deployment, INPUT_IMAGE, PROMPT
        )
    else:
        response_json = call_flux_image_edit(
            FOUNDRY_ENDPOINT, FOUNDRY_API_KEY, FOUNDRY_API_VERSION,
            deployment, INPUT_IMAGE, PROMPT, INPUT_IMAGE_2
        )

    elapsed_sec = time.perf_counter() - start
    print(f"Request completed in {elapsed_sec:.3f}s")


    """
    Save images
    """

    # Create output directory if it doesn't exist
    os.makedirs("generated", exist_ok=True)

    # Sanitize prompt for filename (remove invalid characters)
    import re
    safe_prompt = re.sub(r'[<>:"/\\|?*]', '', PROMPT.replace(' ', '_'))[:50]
    filename_prefix = os.path.join(
        "generated",
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model}_{safe_prompt}"
    )

    try:
        for idx, item in enumerate(response_json['data']):
            # Handle different response formats: 'b64_json' (GPT/FLUX) or 'b64' (some APIs)
            b64_img = item.get('b64_json') or item.get('b64')
            if not b64_img:
                print(f"Error: No image data found in response item {idx}")
                print(f"Item keys: {item.keys()}")
                continue
            filename = f"{filename_prefix}_{idx+1}.png"
            image = Image.open(BytesIO(base64.b64decode(b64_img)))
            image.show()
            image.save(filename)
            print(f"Image saved to: '{filename}'")
    except Exception as e:
        print(f"Error: {e}")
        print(f"Response: {response_json}")