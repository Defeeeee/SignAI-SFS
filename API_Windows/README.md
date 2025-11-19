# SignAI-SFS Windows API Setup

This folder contains the API code optimized for Windows with Python 3.11.9 and NVIDIA GPU (GTX 1650).

## Prerequisites

1.  **Python 3.11.9**: Ensure Python 3.11 is installed and added to PATH.
2.  **NVIDIA Drivers**: Ensure you have the latest drivers for your GTX 1650.
3.  **CUDA Toolkit**: Install CUDA 11.8 or 12.1 (compatible with PyTorch).

## Installation Steps

1.  **Copy Files**:
    Copy the entire `SignAI-SFS` project folder to your Windows machine.

2.  **Create Virtual Environment**:
    Open PowerShell or Command Prompt in the project folder:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install PyTorch with CUDA**:
    Run this command *before* installing other requirements to ensure GPU support:
    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    ```
    *(Note: If you have CUDA 12.1, change `cu118` to `cu121`)*

4.  **Install Dependencies**:
    ```bash
    pip install -r API_Windows/requirements.txt
    ```

## Running the API

1.  **Start the Server**:
    ```bash
    python API_Windows/api.py
    ```

2.  **Verify**:
    -   Wait for the log message: `INFO - Model loaded successfully!`
    -   You should also see: `INFO - CUDA is available. Using GPU.`

## Files in this Folder

-   `api.py`: The FastAPI server adapted for Windows paths and GPU.
-   `video_predict.py`: Prediction logic using `cv2` for Windows.
-   `requirements.txt`: Dependency list for Python 3.11.
