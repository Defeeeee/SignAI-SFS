# Use an official Python runtime as a base image
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app


# Install required Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install torch==1.10.0 torchvision==0.11.1 opencv-python-headless

# Command to run the provided Python script
CMD ["python", "main.py", "--device", "0", "--dataset", "phoenix2014-T", "--phase", "test", "--load-weights", "./best_checkpoints/phoenix2014-T_dev_17.66_test_18.71.pt", "--work-dir", "./work_dir/phoenix"]
