# Active Fire Detection in Landsat-8 Imagery
## Abstract
Based on a large satellite image dataset of patches obtained by directly cutting the original image and converted binary image masks. Then, three existing automatic segmentation algorithms (Kumar and Roy (2018), Murphy et al. (2016), and Schroeder et al. (2016)) were used to label masks in 140,000 datasets and generate fire point labeling. These data sets are divided into training sets, testing sets and verification sets according to 40%, 50% and 10%. Then the intersection algorithm is used to eliminate the noise as much as possible. Finally, the scores of P, R, IoU and F were obtained to reflect the training effect of the model.
## Requirements
Python 3.11.8

PyTorch = 2.4.1+cu121
## Download
Start by downloading the full dataset from Google Cloud Drive. If you don't have that much storage space, you can download the sample dataset and set the `DOWNLOAD_FULL_DATASET` to `False` before running it.
```shell
python download.py
```
Also, click the link to download the pre-training weights.

[Download the weights](https://drive.google.com/file/d/1IL8jKPgyN2d4rSY9mY8Y0YM61bK0tBbj/view?usp=sharing) 
## Unzip
This script will unzip the downloaded files and separete the masks from the image patches. The image patches will be unzipped to `<your-local-repository>/dataset/images/patches` folder, while the masks will be placed in different folders inside `<your-local-repository>/dataset/images/masks`. The masks generated by the mentioned algorithms will be unzipped to `<your-local-repository>/dataset/images/masks/patches`, while the masks produced by their intersection will be unzipped to `<your-local-repository>/dataset/images/masks/intersection`. 
```shell
python unzip.py
```
Similarly, if you downloaded the sample dataset in the first step, set the `DOWNLOAD_FULL_DATASET` to `False` before running it.
## Split
If you decide to train the networks from scratch you will need to separete the samples in three subsets. The samples used for training, test and validation are defined by CSV files inside the `<model path>/dataset` folder. The images_masks.csv file list all the images and corresponding masks for the approach. The `images_train.csv` and `masks_train.csv` files list the files used to train the model, the `*_val.csv` files hold the files for the validation and the `*_test.csv` files have the files used in the test phase. For this purpose you can run:
```shell
python split.py
```
This will create the `images_masks.csv`, the `images_*.csv` and the `masks_*.csv` files. By default the data will be divided in a proportion of 40% for training, 50% for testing and 10% for validation. If you want to change these proportions you need to change the `TRAIN_RATIO`, `TEST_RATIO` and `VALIDATION_RATIO` constants.
## Train
The pre-trained weights are given in the folder along with the code and can be directly referenced. If you want to use pre-trained weights, just change the `WEIGHTS_FILE` path in `inference.py` to skip this section.

If you wish to train a model from scratch you need to run:
```shell
python train.py
```
This will execute all the steps needed to train a new model. This code expects that the samples are in a sibling folder of src named dataset, the images must be in `dataset/images/patches` and the masks in `dataset/masks/patches` and `dataset/masks/intersection` for intersection masks. If you are using other directory to hold your samples you may change the `IMAGES_PATH` and `MASKS_PATH` constants.

The output produced by the training script will be placed at the train_output folder inside the model folder. This repository already includes trained weights inside this folder, so if you retrain the model these weights will be overwritten.

Besides the final weights, this script will save checkpoints every 5 epochs, if you need to resume from a checkpoint you just need to set the constant `INITIAL_EPOCH` with the epoch corresponding to the checkpoint.
## Inference
The testing phase is divided in two main steps. The first one is to pass the `images_test.csv` images through the trained model and save the output as a txt file, where 0 represents background and 1 represents fire. The masks in `masks_test.csv` will also be converted to a txt file. These files will be written in the log folder inside the model folder. The output prediction produced by the CNN will be saved as `det_<image-name>.txt` while the corresponding mask will be saved as `grd_<mask-name>.txt.` To execute this process run:
```shell
python inference.py
```
## Evaluate
If your samples are placed in a diretory other than the default you need to change the constant `IMAGES_PATH` and `MASKS_PATH`. The outputs produced by the CNN are converted to interger through a thresholding process, the default threshold is 0.25. You can change this value in the `TH_FIRE` constant.

After this processes you can start the second step to evaluate your trained model, running:
```shell
python evaluate.py
```
This will show the results from your model.
