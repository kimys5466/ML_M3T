# M3ST

This repository is the official implementation of "M3ST: 3D Medical Image Classifier Using Multi-plane & Multi-slice Swin Transformer, Machine Learning final Project 2023". 

The M3ST model extracts features from MRI images with significantly reduced computational complexity by avoiding the use of 3D CNN. Moreover, it employs the Swin Transformer, which is more suitable for image analysis, to more effectively extract distributed features from MRI images. 

## Requirements
#### Step 1:
To install requirements:
```setup
conda env create -f /path/to/ML_project.yaml
conda activate ML_project
```

#### Step 2:
Create a new empty folder 'data' in this folder.
Download datasets and unzip them to the folder 'data'.
Change the path = "C:/~~" of the main.py file to the location of the file where the data is stored

## Dataset
1. AD_resize
2. CN_resize

Link to download [data](https://drive.google.com/file/d/1HB8YCmZneezeXbMQNH_HHFhKK2vn-Sjf/view?usp=sharing)

## Training and testing

To train and test the M3ST in the paper, run main.py
