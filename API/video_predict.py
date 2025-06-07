import os
import sys
import tempfile
import subprocess
import uuid
import shutil
import requests

# Add project root to sys.path for module import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from prediction.predict import predict_sign

def video_predict(video_url, weights, config='configs/phoenix2014-T.yaml',
                  dict_path='preprocess/phoenix2014-T/gloss_dict.npy', device=None,
                  search_mode='beam', input_size=224, image_scale=1.0):
    """
    Downloads a video from a URL, extracts frames using ffmpeg, predicts the sign language, and returns the predicted string.
    """
    # Automatically detect the best available device
    if device is None:
        import torch
        if torch.cuda.is_available():
            device = 'cuda:0'
            print("Using CUDA for inference.")
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
            print("Using MPS (Apple Silicon) for inference.")
        else:
            device = 'cpu'
            print("Using CPU for inference.")
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, f"{uuid.uuid4()}.mp4")
    frames_dir = os.path.join(temp_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    try:
        # Download video
        with requests.get(video_url, stream=True) as r:
            r.raise_for_status()
            with open(video_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Check if ffmpeg is installed
        if shutil.which('ffmpeg') is None:
            raise RuntimeError("ffmpeg is not installed or not found in PATH. Please install ffmpeg to use this function.")

        # Extract frames using ffmpeg
        ffmpeg_cmd = [
            'ffmpeg', '-i', video_path,
            os.path.join(frames_dir, 'frame_%05d.jpg')
        ]
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Predict sign from frames
        prediction = predict_sign(
            folder=frames_dir,
            weights=os.path.abspath(os.path.join(os.path.dirname(__file__), '../best_checkpoints/phoenix2014-T_dev_17.66_test_18.71.pt')),
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
    prediction = video_predict(video_url, weights='./best_checkpoints/phoenix2014-T_dev_17.66_test_18.71.pt')
    print("Predicted Sign Language:", prediction)
