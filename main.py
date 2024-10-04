import os
from capture_samples import capture_samples
from create_keypoints import create_keypoints

root = os.getcwd()
words_path = os.path.join(root, "action_frames")
data_path = os.path.join(root, "data")
