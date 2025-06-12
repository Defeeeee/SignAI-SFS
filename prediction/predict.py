

import os
import sys
import cv2
import glob
import torch
import argparse
import importlib
import platform
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

# Import necessary modules from the project
import utils
from utils.device import GpuDataParallel
from utils.decode import Decode
from utils.video_augmentation import Compose, CenterCrop, ToTensor

def parse_args():
    """
    Parse command line arguments for the prediction script.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Predict sign language from images in a folder')
    # Required arguments
    parser.add_argument('--folder', type=str, required=True,
                        help='Path to folder containing images (required)')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights file (required)')

    # Optional arguments with defaults
    parser.add_argument('--config', type=str, default='./configs/phoenix2014-T.yaml',
                        help='Path to config file (default: ./configs/phoenix2014-T.yaml)')
    parser.add_argument('--dict_path', type=str, default='./preprocess/phoenix2014-T/gloss_dict.npy',
                        help='Path to gloss dictionary (default: ./datasets_files/PHOENIX-2014-T/info/gloss_dict.npy)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use: cuda:0, mps (for Apple Silicon), or cpu (default: cuda:0)')
    parser.add_argument('--search_mode', type=str, default='beam',
                        help='Search mode for decoding: max or beam (default: beam)')
    parser.add_argument('--input_size', type=int, default=224,
                        help='Input size for the model (default: 224)')
    parser.add_argument('--image_scale', type=float, default=1.0,
                        help='Image scale factor (default: 1.0)')

    return parser.parse_args()

def load_model(args):
    """
    Load the model and its weights from the specified paths.

    1. Loads the configuration from the config file
    2. Loads the gloss dictionary from the dict_path
    3. Imports the model class specified in the config
    4. Creates the model with the appropriate parameters
    5. Loads the weights from the weights file
    6. Moves the model to the specified device
    """
    import yaml
    import importlib

    # Load config file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Load gloss dictionary (mapping between words and indices)
    gloss_dict = np.load(args.dict_path, allow_pickle=True).item()
    num_classes = len(gloss_dict) + 1  # +1 for blank/unknown class

    # Import model class dynamically based on config
    model_path = config.get('model', 'slr_network_multi.SLRModel')
    model_class = import_class(model_path)

    # Use the absolute path for the config file
    slowfast_config = os.path.abspath(os.path.join(os.path.dirname(__file__), '../slowfast_modules/configs', os.path.basename(config.get('slowfast_config', 'SLOWFAST_64x2_R101_50_50.yaml'))))

    # Create model instance with parameters from config
    model = model_class(
        num_classes=num_classes,
        c2d_type=config.get('model_args', {}).get('c2d_type', 'slowfast101'),
        conv_type=config.get('model_args', {}).get('conv_type', 2),
        use_bn=config.get('model_args', {}).get('use_bn', False),
        hidden_size=1024,  # Default value
        gloss_dict=gloss_dict,
        loss_weights=config.get('loss_weights', {}),
        weight_norm=config.get('model_args', {}).get('weight_norm', True),
        share_classifier=config.get('model_args', {}).get('share_classifier', 1),
        load_pkl=False,  # Don't load pre-trained SlowFast weights
        slowfast_config=slowfast_config,
        slowfast_args=config.get('slowfast_args', [])
    )

    # Load weights with appropriate device mapping
    map_location_device = 'mps' if torch.backends.mps.is_available() and args.device == 'mps' else 'cpu'
    state_dict = torch.load(args.weights, map_location=torch.device(map_location_device), weights_only=False)

    # Handle different state dict formats (checkpoint vs. weights-only)
    if 'model_state_dict' in state_dict:
        weights = state_dict['model_state_dict']
    else:
        weights = state_dict

    # Remove module prefix if the model was saved with DataParallel
    weights = {k.replace('.module', ''): v for k, v in weights.items()}

    # Load weights into model
    model.load_state_dict(weights, strict=True)

    # Move model to the specified device
    device = GpuDataParallel()
    # Check if CUDA is available when cuda device is specified
    if 'cuda' in args.device and not torch.cuda.is_available():
        print(f"CUDA is not available. Falling back to CPU.")
        args.device = 'cpu'
    device.set_device(args.device)
    model = device.model_to_device(model)
    model.eval()  # Set model to evaluation mode

    return model, gloss_dict, device

def import_class(name):
    """
    Dynamically import a class from a module.

"""
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod

def process_images(folder_path, input_size=224, image_scale=1.0):
    """
    Process images from a folder for model prediction.

    This function:
    1. Finds all image files in the specified folder
    2. Sorts them by name (important for temporal order)
    3. Reads and converts them to RGB
    4. Applies transformations (center crop and conversion to tensor)
    5. Normalizes the pixel values
    6. Prepares them for model input

   """
    # Get all image files with common extensions
    extensions = ['jpg', 'jpeg', 'png', 'bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, f'*.{ext}')))

    if not image_files:
        raise ValueError(f"No image files found in {folder_path}")

    # Sort images by name to maintain temporal order
    image_files = sorted(image_files)

    # Read and process images
    images = []
    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        # Convert from BGR (OpenCV default) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)

    if not images:
        raise ValueError("No valid images found")

    # Apply transformations using the same pipeline as in training/testing
    transform = Compose([
        CenterCrop(input_size),  # Crop to square from center
        ToTensor(),              # Convert to PyTorch tensor
    ])

    # Apply transformations to images (empty list for labels)
    transformed_images, _ = transform(images, [])

    # Normalize images using the same mean/std as in training
    # (transformed_images - mean) / std
    transformed_images = ((transformed_images.float() / 255.) - 0.45) / 0.225

    # Add batch dimension [batch=1, channels, frames, height, width]
    transformed_images = transformed_images.unsqueeze(0)

    # The model may expect a specific number of frames
    # This can vary based on the model architecture and training data
    # We need to handle both cases: when input has fewer or more frames than expected

    # Determine the expected frame count based on model architecture
    # This could be made configurable in the future
    # For now, we'll use a function to determine the expected frame count
    def get_expected_frames(current_count, alpha=4):
        # If the current count is not a multiple of alpha, pad to make it a multiple
        if current_count % alpha != 0:
            return current_count + (alpha - (current_count % alpha))
        return current_count

    # Get the expected number of frames (ensuring it's a multiple of alpha)
    num_frames = transformed_images.size(1)
    expected_frames = get_expected_frames(num_frames)

    # If padding is needed, add duplicate frames at the end
    if expected_frames > num_frames:
        # Create padding by repeating the last frame
        last_frame = transformed_images[:, -1:, :, :, :]
        padding_needed = expected_frames - num_frames
        padding = last_frame.repeat(1, padding_needed, 1, 1, 1)
        transformed_images = torch.cat([transformed_images, padding], dim=1)
        print(f"Added {padding_needed} padding frames to make frame count ({num_frames}) a multiple of alpha (4)")

    num_frames = transformed_images.size(1)

    # Calculate video length (number of frames)
    video_length = torch.LongTensor([num_frames])

    return transformed_images, video_length

def predict(model, images, video_length, device, gloss_dict, search_mode='max'):
    """
    Run prediction on processed images using the loaded model.

    This function:
    1. Moves the input data to the appropriate device
    2. Runs the model inference
    3. Decodes the model output into gloss predictions

    """
    # Move data to the appropriate device (GPU/CPU)
    images = device.data_to_device(images)
    video_length = device.data_to_device(video_length)

    # Run inference with gradient calculation disabled
    with torch.no_grad():
        ret_dict = model(images, video_length)

    # Create decoder for converting model outputs to gloss predictions
    decoder = Decode(gloss_dict, len(gloss_dict) + 1, search_mode)

    # Decode predictions
    if 'recognized_sents' in ret_dict:
        # Model already decoded the predictions
        predictions = ret_dict['recognized_sents']
    else:
        # Need to manually decode the model outputs
        # Use sequence_logits instead of framewise_features as it's what the model returns
        outputs = ret_dict['sequence_logits'][0]
        # Use the updated feature length from the model's output
        feat_len = ret_dict.get('feat_len', video_length)

        # Debug: Print tensor sizes to help identify the issue
        print(f"Debug: outputs.shape = {outputs.shape}")
        print(f"Debug: feat_len = {feat_len.item()}")

        # Print all keys in ret_dict to see what's available
        print(f"Debug: ret_dict keys = {ret_dict.keys()}")

        # Try using conv_logits instead of sequence_logits
        if 'conv_logits' in ret_dict:
            print(f"Debug: conv_logits[0].shape = {ret_dict['conv_logits'][0].shape}")
            outputs = ret_dict['conv_logits'][0]
            print(f"Debug: Using conv_logits instead of sequence_logits")

        # Ensure tensor sizes match by adjusting feat_len if necessary
        if outputs.size(0) != feat_len.item():
            print(f"Warning: Adjusting feature length from {feat_len.item()} to {outputs.size(0)} to match output tensor size")
            feat_len = torch.tensor([outputs.size(0)], device=feat_len.device)

        predictions = decoder.decode(outputs, feat_len)

    return predictions

def main():
    """
    Main function to run the prediction pipeline.

    This function:
    1. Parses command line arguments
    2. Loads the model and weights
    3. Processes images from the specified folder
    4. Runs prediction on the processed images
    5. Displays the prediction results
    """
    # Parse command line arguments
    args = parse_args()

    # Load model and weights
    print(f"Loading model from {args.weights}...")
    model, gloss_dict, device = load_model(args)

    # Process images from the specified folder
    print(f"Processing images from {args.folder}...")
    images, video_length = process_images(args.folder, args.input_size, args.image_scale)

    # Run prediction using the loaded model
    print("Running prediction...")
    predictions = predict(model, images, video_length, device, gloss_dict, args.search_mode)

    # Print prediction results
    print("\nPrediction Results:")
    print("-" * 50)

    if predictions and len(predictions) > 0:
        for i, pred_seq in enumerate(predictions):
            if pred_seq:
                # Join the predicted words into a sentence
                sentence = " ".join([word for word, _ in pred_seq])
                print(f"Predicted: {sentence}")
            else:
                print("No prediction (empty result)")
    else:
        print("No predictions returned")


def predict_sign(folder, weights, config='./configs/phoenix2014-T.yaml',
                 dict_path='./preprocess/phoenix2014-T/gloss_dict.npy', device='cuda:0',
                 search_mode='beam', input_size=224, image_scale=1.0):
    """
    Predict sign language from a folder of images and return the predicted string.

    """
    # Check if the system is Mac, use absolute paths if it is
    if platform.system() == 'Darwin':  # Darwin is the system name for macOS
        # Use absolute paths for Mac
        base_dir = '/Users/defeee/Documents/GitHub/SignAI-SFS'
        config = os.path.join(base_dir, 'configs/phoenix2014-T.yaml')
        dict_path = os.path.join(base_dir, 'preprocess/phoenix2014-T/gloss_dict.npy')
    args = argparse.Namespace(
        folder=folder,
        weights=weights,
        config=config,
        dict_path=dict_path,
        device=device,
        search_mode=search_mode,
        input_size=input_size,
        image_scale=image_scale
    )
    # Load model and weights
    model, gloss_dict, device_obj = load_model(args)
    # Process images from the specified folder
    images, video_length = process_images(args.folder, args.input_size, args.image_scale)
    # Run prediction using the loaded model
    predictions = predict(model, images, video_length, device_obj, gloss_dict, args.search_mode)
    # Return the first prediction as a string
    if predictions and len(predictions) > 0 and predictions[0]:
        sentence = " ".join([word for word, _ in predictions[0]])
        return sentence
    else:
        return ""

if __name__ == "__main__":
    # Entry point of the script
    try:
        # Set paths based on system
        if platform.system() == 'Darwin':  # Mac
            folder = '/Users/defeee/Documents/GitHub/SignAI-SFS/datasets_files/PHOENIX-2014-T/PHOENIX-2014-T/features/fullFrame-256x256px/test/25October_2010_Monday_tagesschau-17'
            weights = '/Users/defeee/Documents/GitHub/SignAI-SFS/best_checkpoints/phoenix2014-T_dev_17.66_test_18.71.pt'
        else:  # Linux or other
            folder = './datasets_files/PHOENIX-2014-T/PHOENIX-2014-T/features/fullFrame-256x256px/test/25October_2010_Monday_tagesschau-17'
            weights = './best_checkpoints/phoenix2014-T_dev_17.66_test_18.71.pt'

        print(predict_sign(
            folder=folder,
            weights=weights,
        ))
        print("Prediction completed successfully!")
    except Exception as e:
        print(f"Error during prediction: {e}")
