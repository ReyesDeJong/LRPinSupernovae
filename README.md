# Adaptation of LRP framework for tensorflow

An adaptation to the Layer-wise Relevance Propagation (LRP) algorithm developed by **Lapuschkin et al., 2016** on their work **'The LRP Toolbox for Artificial Neural Networks' for the 'Journal of Machine Learning Research'**.


1. Original web page project: [heatmapping.org](http://heatmapping.org)
   
2. Original LRP toolbox repository for raw numpy can be found on [LRP-toolbox](https://github.com/sebastian-lapuschkin/lrp_toolbox)
3. Original LRP toolbox repository for Tensorflow can be found on [LRP-tensorflow](https://github.com/VigneshSrinivasan10/interprettensor)

The LRP algorithm projects a classifier's output to it's input space, by attributing relevance scores to important features of the input. For example, if we have an image as input of a classifier, output score of the model will be projected to the input image as a heatmap that identify relevant pixels for the prediction made. by using the topology of the learned model itself.

This Tensorflow adaptation of LRP provides an implementations of LRP for artificial neural networks (ANN). Specially Multi Layer Perceptrons (MLP) and Convolutional Neuralk Networks (CNN) in the Deep Learning paradigm. 

The focus of this repository is to extend capabilities of original LRP-tensorflow toolbox, by providing examples on various data sets, add relevance propagation through variations of implemented layers, and enable to reproduce results of paper **'Enhanced Rotational Invariant Convolutional Neural Network for Supernovae Detection'** by **Reyes et al., 2018**.

MNIST VIDEO 

SUPERNOVAE VIDEO

### Requirements
    tensorflow >= 1.5.0
    python >= 3
    matplotlib >= 1.3.1
    scikit-image > 0.11.3
    
### What's new

1. Implementation of cyclic pooling layer, like in **'Exploiting cyclic sym-metry in convolutional neural networks'** by **Dieleman et al., 2016**
2. Implementation of rotation layer, like in **'Deep-hits: Rotation invariant convolutional neural network for transient
detection'** by **Cabrera-Vives et al., 2017**
3. Implementation of Batchnormalization layer, although LRP relevance propagation is not fully tested.
4. LRP axamples on HiTS 2013 dataset


## Usage

### 1. Model 

First you must instantiate a model, indicating the layers (modules) in the neural network to be in the form of a Sequence object. A quick way to define a network would be:

        net = sequential32.Sequential([convolution32.Convolution(kernel_size=5, output_depth=32, input_depth=1,
                                   input_dim=28, act ='relu',
                                   stride_size=1, pad='SAME'),
                       maxpool32.MaxPool(),

                       convolution32.Convolution(kernel_size=5,output_depth=64, stride_size=1,
                                   act ='relu', pad='SAME'),
                       maxpool32.MaxPool(),
                       
                       linear32.Linear(1024, act ='relu'),

                       linear32.Linear(10, act ='linear')])

        output = net.forward(input_data)

This way of defining the network, provides a way to iteratively go though all network modules (layes) both in the forward pass and in the backward LRP propagation of relevance though the different layers.
             
### 2. Train the model

This `net` object can then be trained by calling the next method

        trainer = net.fit(output=score, ground_truth=y_, opt_params=learning_rate)
        
`trainer` is an optimizer that by default is defined as Adam with cross entropy. To modify this optimizer, see file `modules/train2.py` or implement your own. 

### 3. LRP - Layer-wise relevance propagation

Compute the relevances of the input pixels towards the prediction by

        relevances = net.lrp(output, lrp_rule, lrp_rule_param)

the different `lrp_rule` variants available are:

        'simple', 'epsilon','flat','ww' and 'alphabeta' 
        
but, we highly recommend usage of `epsilon` or `alphabeta` rules.

The resulting `relevances` is a variable with same dimensions as the input, and in case of a single-channel image, can be visualized as a heatmap. Like:

### 4. Compute relevances every layer backwards from the output to the input  

Follow steps (1) from Features mentioned above.

       relevance_layerwise = []
       R = output
       for layer in net.modules[::-1]:
           R = net.lrp_layerwise(layer, R, 'simple')
           relevance_layerwise.append(R)
<!---


This tensorflow wrapper provides simple and accessible stand-alone implementations of LRP for artificial neural networks.

<img src="doc/images/1.png" width="215" height="215"> <img src="doc/images/2.png" width="215" height="215"> <img src="doc/images/3.png" width="215" height="215"> <img src="doc/images/4.png" width="215" height="215">


    
# Features

## 1. Model 

This TF-wrapper considers the layers in the neural network to be in the form of a Sequence. A quick way to define a network would be

        net = Sequential([Linear(input_dim=784,output_dim=1296, act ='relu', batch_size=FLAGS.batch_size),
                     Linear(1296, act ='relu'), 
                     Linear(1296, act ='relu'),
                     Linear(10, act ='relu'),
                     Softmax()])

        output = net.forward(input_data)
             
## 2. Train the network

This `net` can then be used to propogate and optimize using

        trainer = net.fit(output, ground_truth, loss='softmax_crossentropy', optimizer='adam', opt_params=[FLAGS.learning_rate])

## 3. LRP - Layer-wise relevance propagation

And compute the contributions of the input pixels towards the decision by

        relevance = net.lrp(output, 'simple', 1.0)

the different lrp variants available are:

        'simple'and 'epsilon','flat','ww' and 'alphabeta' 

## 4. Compute relevances every layer backwards from the output to the input  

Follow steps (1) from Features mentioned above.

       relevance_layerwise = []
       R = output
       for layer in net.modules[::-1]:
           R = net.lrp_layerwise(layer, R, 'simple')
           relevance_layerwise.append(R)
           


# LRP for a pretrained model

Follow steps (1) and (3) from Features mentioned above.


# The LRP Toolbox Paper

When using (any part) of this wrapper, please cite [our paper](http://jmlr.org/papers/volume17/15-618/15-618.pdf)

    @article{JMLR:v17:15-618,
        author  = {Sebastian Lapuschkin and Alexander Binder and Gr{{\'e}}goire Montavon and Klaus-Robert M{{{\"u}}}ller and Wojciech Samek},
        title   = {The LRP Toolbox for Artificial Neural Networks},
        journal = {Journal of Machine Learning Research},
        year    = {2016},
        volume  = {17},
        number  = {114},
        pages   = {1-5},
        url     = {http://jmlr.org/papers/v17/15-618.html}
    }


    
# Misc

For further research and projects involving LRP, visit [heatmapping.org](http://heatmapping.org)
   
-->
