import os
import sys
import subprocess
import glob
import shutil
import tempfile

def frames_to_video(frames_dir, output_path, fps=25):
    """
    Converts a folder of frames to an mp4 video using ffmpeg.
    Supports frames named like images001.png, images002.png, etc.

    Args:
        frames_dir (str): Path to the folder containing frames
        output_path (str): Path to save the output mp4 video
        fps (int): Frames per second for the output video
    """
    # Check if ffmpeg is installed
    if not shutil.which('ffmpeg'):
        raise RuntimeError("ffmpeg is not installed or not found in PATH.")

    # Find all png frames and sort them
    frame_files = sorted(glob.glob(os.path.join(frames_dir, '*.png')))
    if not frame_files:
        print(f"No frames found in {frames_dir} (expected .png files)")
        sys.exit(1)

    # Create a temp dir for symlinks
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, src in enumerate(frame_files, 1):
            dst = os.path.join(tmpdir, f'frame_{idx:05d}.png')
            os.symlink(os.path.abspath(src), dst)
        input_pattern = os.path.join(tmpdir, 'frame_%05d.png')
        cmd = [
            'ffmpeg',
            '-framerate', str(fps),
            '-i', input_pattern,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-y',  # Overwrite output file if exists
            output_path
        ]
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert a folder of frames to an mp4 video using ffmpeg.")
    parser.add_argument('frames_dir', type=str, help='Path to the folder containing frames')
    parser.add_argument('--output', type=str, default=None, help='Output mp4 file path (default: output.mp4 in frames_dir)')
    parser.add_argument('--fps', type=int, default=25, help='Frames per second (default: 25)')
    args = parser.parse_args()

    # Set output path
    output_path = args.output or os.path.join(args.frames_dir, 'output.mp4')
    frames_to_video(args.frames_dir, output_path, fps=args.fps)
    print(f"Video saved to {output_path}")

