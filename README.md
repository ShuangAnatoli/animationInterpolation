# Video Frame Interpolation Program

Example Video : [Video](https://drive.google.com/file/d/1P6ZwqpFtvcbqmhZ6JRDnf7xgvDuVwW2w/view?usp=sharing)
For the full paper: [Paper](https://drive.google.com/file/d/1EESd81NSs93OJYb42DartC5udTlOShRp/view?usp=sharing)

## Installation
```
pip install opencv-python 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Run the Program
```
py combined.py
```
This will load the model from checkpoints.

## Evaluate
```
py generate_eval.py
```
This generates images for evaluation.

```
py eval.py
```
This evaluates the generated images.

## Test
```
py test.py
```
Add frames in test_frames folder.

```
py video.py
```
This combines those 3 frames into .mp4 format.

## Dataset
Dataset is available at: [Google Drive Link](https://drive.google.com/file/d/1vyu_ePFN9sFjqxc-sPdSWuSCLnWFVUT7/view?usp=sharing)
