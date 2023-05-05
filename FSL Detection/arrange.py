# arranges the videos in the signed_videos folder into a format that can be used by the model from 1.mp4-n.mp4, where n is the number of videos in the folder.

import os 
input_dir = 'signed_videos'

# Loop through subdirectories
for subdir in os.listdir(input_dir):
    subdir_path = os.path.join(input_dir, subdir)
    if os.path.isdir(subdir_path):
        # Get list of mp4 files
        mp4_files = [f for f in os.listdir(subdir_path) if f.endswith('.mp4')]
        # Sort files by name
        mp4_files.sort()
        # Rename files
        for i, filename in enumerate(mp4_files):
            old_path = os.path.join(subdir_path, filename)
            new_path = os.path.join(subdir_path, '{}.mp4'.format(i))
            os.rename(old_path, new_path)