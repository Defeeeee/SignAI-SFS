import os
import tempfile
import uuid
import shutil
import requests
import cv2
import logging
from .model import get_model_components, load_model_global
from .config import get_model_args

# Add project root to sys.path for module import
import sys
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from prediction.predict import predict_sign

logger = logging.getLogger(__name__)

def video_predict(video_url, search_mode='beam', input_size=224, image_scale=1.0):
    """
    Downloads a video from a URL, extracts frames using cv2 (in memory), predicts the sign language, and returns the predicted string.
    """
    model, gloss_dict, device = get_model_components()
    
    # Automatically detect the best available device if not provided/loaded
    if model is None:
        logger.info("Model not loaded, attempting lazy load...")
        load_model_global()
        model, gloss_dict, device = get_model_components()
            
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, f"{uuid.uuid4()}.mp4")

    try:
        # Download video
        with requests.get(video_url, stream=True) as r:
            r.raise_for_status()
            with open(video_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Define frame generator
        def frame_generator():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video file {video_path}")
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    yield frame
            finally:
                cap.release()

        frames = frame_generator()

        # Predict sign from frames
        # If model instance is provided, use it directly
        if model is not None and gloss_dict is not None and device is not None:
            from prediction.predict import process_images, predict
            
            # Process images (in memory)
            images, video_length = process_images(frames, input_size, image_scale)
            
            # Run prediction
            predictions = predict(model, images, video_length, device, gloss_dict, search_mode)
            
            # Return the first prediction as a string
            if predictions and len(predictions) > 0 and predictions[0]:
                sentence = " ".join([word for word, _ in predictions[0]])
                return sentence
            else:
                return ""
        else:
            # Fallback to loading model (slower) - this path should rarely be hit if load_model_global works
            args = get_model_args()
            prediction = predict_sign(
                folder=frames, # Pass list of frames directly
                weights=args.weights,
                config=args.config,
                dict_path=args.dict_path,
                device=args.device,
                search_mode=search_mode,
                input_size=input_size,
                image_scale=image_scale
            )
            return prediction
    finally:
        shutil.rmtree(temp_dir)
