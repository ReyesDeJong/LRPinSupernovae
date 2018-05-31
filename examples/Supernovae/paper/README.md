# USAGE

This folder contains two example for LRP on HiTS 2013 dataset, one for the original DeepHiTS model and other for the enhanced model proposed in the repo's paper. HiTS dataset in `.tfrecords` format will be soon publicly available. Each `*.ipynb` jupyter notebook contains comments and a detailed explanation of each code block.

### 1. Training

Each example have a `*_trainning.ipynb` notebook which is used to train the model and store respective parameters on a `weights` folder. To train each model and save its parameters just download the data sets and execute every block of the respective notebook.

Training process can take up to 3 hours on a GEFORCE GTX 1080Ti GPU.

### 2. LRP visualization

After training a model, weights folder should be generated, now you can open `Pretrained_*_LRP_Example.ipynb` notebook for the respective model and execute every code block to generate LRP visualizations. Every block code is explained in detail, especially those who talk about LRP parameters and usage. 

An example of what you can get:

![alt text](https://github.com/ReyesDeJong/LRPpaper/blob/master/doc/images/SNtwo.png)
