# USAGE

This folder contains an example for LRP on MNIST 2013 dataset. MNIST dataset is downloaded automatically by the `*.ipynb` jupyter notebook files. Each `*.ipynb` jupyter notebook contains comments and a detailed explanation of each code block.
  
- numbers_images: folder with `*.npy` numpy files of different MNIST images.
- weights_MNIST_MaxPool: folder with `*.npy` numpy files of parameters for the model trained with `MNIST_trainning.ipynb` and that can be used on LRP examples of `PreTrained_*_Example.ipynb` files.
- `MNIST_trainning.ipynb`: jupyter notebook to train an MNIST CNN model.
- `PreTrained_MNIST_LRP_Example.ipynb`: jupyter notebook to run a simple LRP visualization on pretrained MNIST CNN model.
- `PreTrained_MNIST_LRP_Rules_Comparison_Example.ipynb`: jupyter notebook to run a different rules LRP visualization on pretrained MNIST CNN model.

### 1. Training

`*_trainning.ipynb` notebook is used to train the model and store respective parameters on a `weights` folder. Execute every block of the notebook.

### 2. LRP visualization

After training a model, weights folder should be generated, now you can open any `Pretrained_*_Example.ipynb` notebook, select path to model parameters, and execute every code block to generate LRP visualizations. Every block code is explained in detail, especially those who talk about LRP parameters and usage. 

An example of what you can get:

![alt text](https://github.com/ReyesDeJong/LRPpaper/blob/master/doc/images/40.png)
