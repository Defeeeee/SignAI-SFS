import os
import sys
import tempfile
import subprocess
import uuid
import shutil
import requests
import platform
import cv2

# Add project root to sys.path for module import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from prediction.predict import predict_sign

def video_predict(video_url, weights=None, config='./configs/phoenix2014-T.yaml',
                  dict_path='./preprocess/phoenix2014-T/gloss_dict.npy', device=None,
                  search_mode='beam', input_size=224, image_scale=1.0, 
                  model_instance=None, gloss_dict_instance=None, device_instance=None):
    """
    Downloads a video from a URL, extracts frames using cv2 (in memory), predicts the sign language, and returns the predicted string.
    """
    # Check if the system is Mac, use absolute paths if it is
    if platform.system() == 'Darwin':  # Darwin is the system name for macOS
        # Use absolute paths for Mac
        base_dir = '/Users/defeee/Documents/GitHub/SignAI-SFS'
        config = os.path.join(base_dir, 'configs/phoenix2014-T.yaml')
        dict_path = os.path.join(base_dir, 'preprocess/phoenix2014-T/gloss_dict.npy')
        
    # Automatically detect the best available device if not provided
    if device is None and device_instance is None:
        import torch
        if torch.cuda.is_available():
            device = 'cuda:0'
            # print("Using CUDA for inference.")
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
            # print("Using MPS (Apple Silicon) for inference.")
        else:
            device = 'cpu'
            # print("Using CPU for inference.")
            
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
        if model_instance is not None and gloss_dict_instance is not None and device_instance is not None:
            from prediction.predict import process_images, predict
            
            # Process images (in memory)
            images, video_length = process_images(frames, input_size, image_scale)
            
            # Run prediction
            predictions = predict(model_instance, images, video_length, device_instance, gloss_dict_instance, search_mode)
            
            # Return the first prediction as a string
            if predictions and len(predictions) > 0 and predictions[0]:
                sentence = " ".join([word for word, _ in predictions[0]])
                return sentence
            else:
                return ""
        else:
            # Fallback to loading model (slower)
            prediction = predict_sign(
                folder=frames, # Pass list of frames directly
                weights=weights,
                config=config,
                dict_path=dict_path,
                device=device,
                search_mode=search_mode,
                input_size=input_size,
                image_scale=image_scale
            )
            return prediction
    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    # Example usage
    #video_url="https://res.cloudinary.com/dv4xloi62/video/upload/v1749229375/npmgyj9rjyurw3xfoeqk.mp4"
    #video_url = "https://res.cloudinary.com/dv4xloi62/video/upload/v1749250766/q8hl3mfw7kef2ncqsmab.mp4"
    video_url = "https://res.cloudinary.com/dv4xloi62/video/upload/v1749217872/t9zjtrvlcnpkhwhaazrg.mp4"

    # Set weights path based on system
    if platform.system() == 'Darwin':  # Mac
        weights = '/Users/defeee/Documents/GitHub/SignAI-SFS/best_checkpoints/phoenix2014-T_dev_17.66_test_18.71.pt'
    else:  # Linux or other
        weights = './best_checkpoints/phoenix2014-T_dev_17.66_test_18.71.pt'

    prediction = video_predict(video_url, weights=weights)
    print("Predicted Sign Language:", prediction)
