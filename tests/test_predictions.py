"""
Test file for SignAI-SFS prediction functionality.

This file contains tests for the video_predict and predict_sign functions,
verifying that they return the expected outputs for specific inputs.

To run these tests:
1. Make sure pytest is installed:
   pip install pytest

2. Run the tests from the project root directory:
   pytest tests/test_predictions.py -v

Note: These tests require an internet connection to download the test videos
and ffmpeg to be installed for video frame extraction.
"""

import os
import sys
import platform

# Note: You may need to install pytest if it's not already installed
# pip install pytest
import pytest

# Add project root to sys.path for module import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from API.video_predict import video_predict
from prediction.predict import predict_sign

# Set paths based on system
if platform.system() == 'Darwin':  # Mac
    BASE_DIR = '/Users/defeee/Documents/GitHub/SignAI-SFS'
    WEIGHTS = os.path.join(BASE_DIR, 'best_checkpoints/phoenix2014-T_dev_17.66_test_18.71.pt')
    TEST_FOLDER = os.path.join(BASE_DIR, 'datasets_files/PHOENIX-2014-T/PHOENIX-2014-T/features/fullFrame-256x256px/test/25October_2010_Monday_tagesschau-17')
else:  # Linux or other
    BASE_DIR = '.'
    WEIGHTS = './best_checkpoints/phoenix2014-T_dev_17.66_test_18.71.pt'
    TEST_FOLDER = './datasets_files/PHOENIX-2014-T/PHOENIX-2014-T/features/fullFrame-256x256px/test/25October_2010_Monday_tagesschau-17'

# Test URLs from video_predict.py
FIRST_VIDEO_URL = "https://res.cloudinary.com/dv4xloi62/video/upload/v1749250766/q8hl3mfw7kef2ncqsmab.mp4"
SECOND_VIDEO_URL = "https://res.cloudinary.com/dv4xloi62/video/upload/v1749217872/t9zjtrvlcnpkhwhaazrg.mp4"

# Expected outputs
EXPECTED_FIRST_VIDEO = "DIENSTAG BESONDERS REGION MEHR FREUNDLICH LANG ABER AUCH DABEI SCHAUER"
EXPECTED_SECOND_VIDEO = "JETZT WETTER WIE-AUSSEHEN MORGEN SAMSTAG ZWEITE APRIL"
EXPECTED_TEST_FOLDER = "REGEN SCHNEE REGION VERSCHWINDEN NORD REGEN KOENNEN"

@pytest.mark.parametrize(
    "video_url,expected_output",
    [
        (FIRST_VIDEO_URL, EXPECTED_FIRST_VIDEO),
        (SECOND_VIDEO_URL, EXPECTED_SECOND_VIDEO),
    ]
)
def test_video_predict(video_url, expected_output):
    """
    Test video_predict function with different video URLs.

    This test verifies that the video_predict function returns the expected
    output for the given video URLs.
    """
    # Skip test if running in CI environment without ffmpeg
    if os.environ.get('CI') == 'true' and not os.system('which ffmpeg') == 0:
        pytest.skip("ffmpeg not available in CI environment")

    # Run prediction
    prediction = video_predict(
        video_url=video_url,
        weights=WEIGHTS,
        device='cpu'  # Use CPU for testing to ensure consistent results
    )

    # Assert that the prediction matches the expected output
    assert prediction == expected_output, f"Expected '{expected_output}', but got '{prediction}'"

def test_predict_sign_from_folder():
    """
    Test predict_sign function with a folder of images.

    This test verifies that the predict_sign function returns the expected
    output for the given folder of images.
    """
    # Run prediction
    prediction = predict_sign(
        folder=TEST_FOLDER,
        weights=WEIGHTS,
        device='cpu'  # Use CPU for testing to ensure consistent results
    )

    # Assert that the prediction matches the expected output
    assert prediction == EXPECTED_TEST_FOLDER, f"Expected '{EXPECTED_TEST_FOLDER}', but got '{prediction}'"
