# Running the Sign Language Recognition Model on Custom Videos

This guide explains how to use the `run_on_custom_video.py` script to run the pretrained sign language recognition model on your own videos.

## Prerequisites

- Python 3.6+
- PyTorch
- OpenCV
- NumPy
- A pretrained model checkpoint (e.g., `phoenix2014-T_dev_17.66_test_18.71.pt`)

## Usage

```bash
python run_on_custom_video.py --video /path/to/your/video.mp4 --weights /path/to/checkpoint.pt --device 0
```

### Command Line Arguments

- `--video`: Path to your video file (required)
- `--weights`: Path to the model weights file (required)
- `--device`: Device to use for inference (default: '0')
  - Use '0' for GPU 0, '1' for GPU 1, etc.
  - Use 'cpu' for CPU inference
- `--dataset`: Dataset name for loading the configuration (default: 'phoenix2014-T')
- `--input_size`: Input size for the model (default: 224)
- `--image_scale`: Image scale factor (default: 1.0)

## Example

```bash
python run_on_custom_video.py --video ./my_sign_video.mp4 --weights ./best_checkpoints/phoenix2014-T_dev_17.66_test_18.71.pt --device 0
```

## Video Requirements

For best results, your video should:

1. Be centered on the person signing
2. Have good lighting
3. Have a clean background
4. Be recorded at a reasonable frame rate (25-30 fps)
5. Show the full upper body of the person signing

### Frame Count Considerations

The SlowFast network used for sign language recognition processes videos at two different frame rates (a "slow" pathway and a "fast" pathway). For optimal performance:

- The script automatically pads videos to ensure the frame count is divisible by 4 (the default alpha value)
- If your video has a frame count that's not divisible by 4, the script will add copies of the last frame to make it divisible
- You'll see a message indicating how many padding frames were added

## Output

The script will output the recognized sign language gloss sequence to the console. For example:

```
Recognized sign language:
HELLO MY NAME J-O-H-N NICE MEET YOU
```

## Troubleshooting

If you encounter any issues:

1. Make sure your video file exists and can be opened by OpenCV
2. Check that the model weights file exists and is correctly specified
3. Ensure you have the correct dataset configuration file in the `configs` directory
4. If using GPU, make sure CUDA is properly installed and configured
5. Try using CPU inference if GPU inference fails

### Understanding the Output

If you see a lot of `<UNK>` tokens in the output, this means the model couldn't recognize any known signs in your video. This could be due to:

1. The video doesn't contain signs that the model was trained to recognize
2. The signs are performed differently than in the training data
3. The lighting, background, or camera angle is significantly different from the training data
4. The video quality is too low

The model was trained on specific sign language datasets (like PHOENIX2014-T) which contain a limited vocabulary of signs. If your signs are not in this vocabulary, they will be recognized as `<UNK>` (unknown).

For best results:
- Use signs that are part of the dataset the model was trained on
- Ensure good lighting and a clean background
- Position the camera to clearly show your upper body and hands
- Make signs clearly and at a moderate pace

### Apple Silicon (M1/M2/M3) Users

The script automatically sets the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` when running on Apple Silicon devices. This allows operations not supported by MPS (Metal Performance Shaders) to fall back to CPU automatically.

If you encounter errors related to unsupported operations on MPS, you can manually set this environment variable before running the script:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python run_on_custom_video.py --video /path/to/your/video.mp4 --weights /path/to/checkpoint.pt --device 0
```

Alternatively, you can force CPU-only execution by setting the device to 'cpu':

```bash
python run_on_custom_video.py --video /path/to/your/video.mp4 --weights /path/to/checkpoint.pt --device cpu
```
