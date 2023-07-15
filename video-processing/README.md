# Senyas: Filipino Sign Language Translator solution.

## Purpose
We implement a Filipino Sign Language Recognition model for both static and dynamic hand gestures using MediaPipe extracted landmarks. The files in this repository are separated into three categories: `video processing`, `model processing`, and `web application`. 

## We capture videos for the dataset and then process it. The videos are processed and then the landmarks are extracted.
1. `arrange_files.ipynb` is a notebook that manipulates directories and files for convenience of the user. Example is by concatenating old video files to newer ones, and concatenating MP_Data folders together etc.
2. `video_capture.ipynb` captures 30 frame videos N times for convenience of the user. This notebook explains the directory format followed by the project.
3. `video_preprocess.ipynb`: In order to implement a Senyas, we first preprocess the videos via squeezing/augmenting the video inputs into 30 frames.
4. `video_landmark_extraction.ipynb`: The video inputs are then sent into OpenCV to capture frames and into the MediaPipe Holistic pipeline for landmark extraction for each frame. 

## License
Senyas: FSL Translator code and model weights are released under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for additional details.

## Citing Senyas
If you find this repository useful, please consider giving a star :star: and citation
