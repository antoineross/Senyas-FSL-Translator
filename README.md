# Senyas: Filipino Sign Language Translator solution.

## Purpose
We implement a Filipino Sign Language Recognition model for both static and dynamic hand gestures using MediaPipe extracted landmarks. The files in this repository are separated into three categories: `video processing`, `model processing`, and `web application`. 

## Model and Video Processing Usage
First install pytorch 1.13+ and other 3rd party dependencies.

```shell
conda create --name fsl-translator python=3.9 -y
conda activate fsl-translator

pip install -r requirements.txt
```

Under `video processing`:
1. `arrange_files.ipynb` is a notebook that manipulates directories and files for convenience of the user. Example is by concatenating old video files to newer ones, and concatenating MP_Data folders together etc.
2. `video_capture.ipynb` captures 30 frame videos N times for convenience of the user. This notebook explains the directory format followed by the project.
3. `video_preprocess.ipynb`: In order to implement a Senyas, we first preprocess the videos via squeezing/augmenting the video inputs into 30 frames.
4. `video_landmark_extraction.ipynb`: The video inputs are then sent into OpenCV to capture frames and into the MediaPipe Holistic pipeline for landmark extraction for each frame. 

Under `model processing`:
1. `model_training.ipynb`: The extracted landmarks are saved in MP_Data folder which is used as input to the model architecture. This model can be evaluated in this notebook using a confusion matrix and accuracy score.
2. `model_testing.ipynb`: This notebook allows the user to test the saved model in `model_training.ipynb`.

## Web Application Usage
You can visit the official website at [Official Website](https://senyas.vercel.app/). 

You can also run the web application locally. Check the official documentation for more details: [Official Documentation](https://senyas.vercel.app/docs/intro)

## License

Senyas: FSL Translator code and model weights are released under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for additional details.

## Citing Senyas

If you find this repository useful, please consider giving a star :star: and citation
