
# Bare Minimum Requirements to Run SignAI-SFS

Based on the codebase analysis, here are the minimum requirements to run the SignAI-SFS system:

## 1. Python Environment
- Python 3.8
- PyTorch 1.13
- (Optional) ctcdecode 1.0.3 for more efficient beam search decoding. If not available, a custom implementation will be used.
- Additional dependencies from requirements.txt:
  ```
  numpy>=1.20.3
  opencv-python==4.5.5.64
  pandas==1.3.4
  Pillow==9.4.0
  PyYAML==6.0
  scipy==1.7.1
  six==1.16.0
  tqdm==4.62.3
  fvcore
  ```

## 2. Required Files
- **Model weights**: Pre-trained model file (e.g., `./best_checkpoints/phoenix2014-T_dev_17.66_test_18.71.pt`)
- **SlowFast checkpoint**: `./ckpt/SLOWFAST_64x2_R101_50_50.pkl` (can be downloaded with `wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/pretrain/SLOWFAST_64x2_R101_50_50.pkl`)
- **Configuration files**: 
  - `./configs/phoenix2014-T.yaml`
  - `./slowfast_modules/configs/SLOWFAST_64x2_R101_50_50.yaml`
- **Dictionary files**:
  - `./preprocess/phoenix2014-T/gloss_dict.npy` or
  - `./datasets_files/PHOENIX-2014-T/info/gloss_dict.npy`

## 3. Input Data
- A folder containing sign language images
- Images should be named in a way that they are sorted correctly when using the `sorted()` function
  - Example: img_001.jpg, img_002.jpg, img_003.jpg, etc.

## 4. Hardware Requirements
- **Minimum**: CPU-only system
- **Recommended**: CUDA-compatible GPU
- **Also supported**: Apple Silicon (M1/M2/M3) with MPS

## 5. Basic Command to Run
```bash
python predict.py --folder ./path/to/images --weights ./best_checkpoints/phoenix2014-T_dev_17.66_test_18.71.pt
```

## 6. Device-Specific Commands
- **For CPU-only systems**:
  ```bash
  python predict.py --folder ./path/to/images --weights ./best_checkpoints/phoenix2014-T_dev_17.66_test_18.71.pt --device cpu
  ```

- **For systems with CUDA GPU**:
  ```bash
  python predict.py --folder ./path/to/images --weights ./best_checkpoints/phoenix2014-T_dev_17.66_test_18.71.pt --device cuda:0
  ```

- **For Apple Silicon (M1/M2/M3) Macs**:
  ```bash
  python predict.py --folder ./path/to/images --weights ./best_checkpoints/phoenix2014-T_dev_17.66_test_18.71.pt --device mps
  ```

## 7. Advanced Options
Additional parameters that can be customized:
- `--config`: Path to config file (default: ./configs/phoenix2014-T.yaml)
- `--dict_path`: Path to gloss dictionary
- `--search_mode`: Search mode for decoding (max or beam, default: beam)
- `--input_size`: Input size for the model (default: 224)
- `--image_scale`: Image scale factor (default: 1.0)

## 8. Example Full Command with All Options
```bash
python predict.py \
  --folder ./example_images \
  --weights ./best_checkpoints/phoenix2014-T_dev_17.66_test_18.71.pt \
  --config ./configs/phoenix2014-T.yaml \
  --dict_path ./preprocess/phoenix2014-T/gloss_dict.npy \
  --device cuda:0 \
  --search_mode beam \
  --input_size 224 \
  --image_scale 1.0
```
