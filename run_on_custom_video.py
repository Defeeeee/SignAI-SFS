import os
# Set environment variable for MPS fallback if using Apple Silicon
# This must be done before importing torch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import sys
import torch
import argparse
import numpy as np
import yaml
import importlib
from utils.device import GpuDataParallel
from utils import video_augmentation

def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod

def load_model(args):
    print("Loading model")
    # Load the gloss dictionary
    gloss_dict = np.load(args.dict_path, allow_pickle=True).item()

    # Initialize the model
    model_class = import_class(args.model)
    model = model_class(
        **args.model_args,
        gloss_dict=gloss_dict,
        loss_weights=args.loss_weights,
        load_pkl=False,
        slowfast_config=args.slowfast_config,
        slowfast_args=args.slowfast_args
    )

    # Load weights
    map_location_device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    state_dict = torch.load(args.weights, map_location=torch.device(map_location_device), weights_only=False)
    weights = {k.replace('.module', ''): v for k, v in state_dict['model_state_dict'].items()}

    # Try to load weights with strict=False to allow for mismatches in the classifier layers
    try:
        model.load_state_dict(weights, strict=False)
        print("Loaded weights with strict=False. Some weights may not have been loaded.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Trying to load with strict=False and ignoring classifier layers...")

        # Filter out classifier layers that might have different shapes
        filtered_weights = {k: v for k, v in weights.items() if not (k.startswith('classifier') or k.startswith('conv1d.fc'))}
        model.load_state_dict(filtered_weights, strict=False)
        print("Loaded weights with classifier layers excluded.")

    # Move model to device
    device = GpuDataParallel()
    device.set_device(args.device)
    model = device.model_to_device(model)
    model.eval()

    print("Loading model finished.")
    return model, device, gloss_dict

def process_video(video_path, input_size=224, image_scale=1.0, alpha=4):
    print(f"Processing video: {video_path}")

    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist.")
        return None, None

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}.")
        return None, None

    # Read frames from the video
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if not frames:
        print(f"Error: No frames could be read from {video_path}.")
        return None, None

    original_frame_count = len(frames)
    print(f"Read {original_frame_count} frames from video.")

    # Ensure the number of frames is divisible by alpha (usually 4)
    # This is important for the SlowFast network which has two pathways with different frame rates
    remainder = original_frame_count % alpha
    if remainder != 0:
        # Pad with copies of the last frame to make divisible by alpha
        padding_needed = alpha - remainder
        print(f"Padding video with {padding_needed} frames to make frame count divisible by {alpha}")
        last_frame = frames[-1]
        for _ in range(padding_needed):
            frames.append(last_frame.copy())

    print(f"Final frame count: {len(frames)} (divisible by {alpha})")

    # Apply the same transformations as in the test phase
    transform = video_augmentation.Compose([
        video_augmentation.CenterCrop(input_size),
        video_augmentation.Resize(image_scale),
        video_augmentation.ToTensor(),
    ])

    # Apply transformations
    processed_frames, _ = transform(frames, [])

    # Add batch dimension
    processed_frames = processed_frames.unsqueeze(0)

    # Calculate frame length
    frame_length = torch.LongTensor([len(frames)])

    return processed_frames, frame_length

def run_inference(model, device, video_tensor, frame_length):
    print("Running inference")

    with torch.no_grad():
        # Move data to device
        video_tensor = device.data_to_device(video_tensor)
        frame_length = device.data_to_device(frame_length)

        # Run the model
        ret_dict = model(video_tensor, frame_length)

    return ret_dict

def decode_output(ret_dict, gloss_dict):
    # Get the recognized sentences
    recognized_sents = ret_dict['recognized_sents']

    # Create a reverse dictionary to map indices to glosses
    reverse_dict = {v[0]: k for k, v in gloss_dict.items()}

    # Decode the recognized sentences
    decoded_sents = []
    for sent in recognized_sents:
        decoded_sent = []
        for word in sent:
            if word[0] in reverse_dict:
                decoded_sent.append(reverse_dict[word[0]])
            else:
                decoded_sent.append("<UNK>")
        decoded_sents.append(" ".join(decoded_sent))

    return decoded_sents

def main():
    parser = argparse.ArgumentParser(description='Run sign language recognition on a custom video')
    parser.add_argument('--video', type=str, required=True, help='Path to the video file')
    parser.add_argument('--weights', type=str, required=True, help='Path to the model weights')
    parser.add_argument('--device', type=str, default='0', help='Device to use (e.g., "0" for GPU 0, "cpu" for CPU)')
    parser.add_argument('--dataset', type=str, default='phoenix2014-T', help='Dataset name (for loading config)')
    parser.add_argument('--input_size', type=int, default=224, help='Input size for the model')
    parser.add_argument('--image_scale', type=float, default=1.0, help='Image scale factor')
    args = parser.parse_args()

    # Load dataset config
    with open(f"./configs/{args.dataset}.yaml", 'r') as f:
        dataset_info = yaml.load(f, Loader=yaml.FullLoader)

    # Load model config
    with open("./configs/baseline.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Update args with config
    args.dict_path = dataset_info['dict_path']
    args.model = config['model']
    args.model_args = config['model_args']

    # Load the gloss dictionary to get the correct number of classes
    gloss_dict = np.load(args.dict_path, allow_pickle=True).item()
    args.model_args['num_classes'] = len(gloss_dict) + 1

    args.loss_weights = config['loss_weights']
    args.slowfast_config = config['slowfast_config']

    # Convert slowfast_args to list format
    slowfast_args = []
    if 'slowfast_args' in config:
        for key, value in config['slowfast_args'].items():
            slowfast_args.append(key)
            slowfast_args.append(value)
    args.slowfast_args = slowfast_args

    # Load the model
    model, device, gloss_dict = load_model(args)

    # Get the alpha value from the SlowFast config
    alpha = 4  # Default value
    try:
        with open(f"./slowfast_modules/configs/{args.slowfast_config}", 'r') as f:
            sf_config = yaml.load(f, Loader=yaml.FullLoader)
            alpha = sf_config.get('SLOWFAST', {}).get('ALPHA', 4)
            print(f"Using alpha value of {alpha} from SlowFast config")
    except Exception as e:
        print(f"Could not read alpha from config, using default value of {alpha}: {e}")

    # Process the video
    video_tensor, frame_length = process_video(args.video, args.input_size, args.image_scale, alpha=alpha)
    if video_tensor is None:
        return

    # Run inference
    ret_dict = run_inference(model, device, video_tensor, frame_length)

    # Decode the output
    decoded_sents = decode_output(ret_dict, gloss_dict)

    # Print the results
    print("\nRecognized sign language:")
    for sent in decoded_sents:
        print(sent)

if __name__ == '__main__':
    main()
