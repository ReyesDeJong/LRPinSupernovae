import numpy,PIL,PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

lowest = -1.0
highest = 1.0

# --------------------------------------
# Sampling data
# --------------------------------------

def getMNISTsample(N=12,seed=None,path=''):

	fx = '%s/t10k-images.idx3-ubyte'%path
	ft = '%s/t10k-labels.idx1-ubyte'%path

	X  = numpy.fromfile(open(fx),dtype='ubyte',count=16+784*10000)[16:].reshape([10000,784])
	T  = numpy.fromfile(open(ft),dtype='ubyte',count=8+10000)[8:]
	T  = (T[:,numpy.newaxis]  == numpy.arange(10))*1.0

	if seed==None: seed=numpy.random
	else: seed=numpy.random.mtrand.RandomState(seed)

	R = seed.randint(0,len(X),[N])
	X,T = X[R],T[R]

	return X/255.0*(highest-lowest)+lowest,T

# --------------------------------------
# Color maps ([-1,1] -> [0,1]^3)
# --------------------------------------

def heatmap(x):

	x = x[...,numpy.newaxis]

	r = 0.9 - numpy.clip(x-0.3,0,0.7)/0.7*0.5
	g = 0.9 - numpy.clip(abs(x)-0.0,0,0.3)/0.3*0.5 - numpy.clip(x-0.3,0,0.7)/0.7*0.4
	b = 0.9 - numpy.clip(x-0.0,0,0.3)/0.3*0.5 - numpy.clip(x-0.3,0,0.7)/0.7*0.4

	return numpy.concatenate([r,g,b],axis=-1)

def graymap(x):

	x = x[...,numpy.newaxis]
	return numpy.concatenate([x,x,x],axis=-1)*0.5+0.5

# --------------------------------------
# Visualizing data
# --------------------------------------

def visualize(x,colormap,name):

	N = len(x); assert(N<=16)

	x = colormap(x/numpy.abs(x).max())

	# Create a mosaic and upsample
	x = x.reshape([1,N,28,28,3])
	x = numpy.pad(x,((0,0),(0,0),(2,2),(2,2),(0,0)),'constant',constant_values=1)
	x = x.transpose([0,2,1,3,4]).reshape([1*32,N*32,3])
	x = numpy.kron(x,numpy.ones([2,2,1]))

	PIL.Image.fromarray((x*255).astype('byte'),'RGB').save(name)

##Generate other classes mask, for LRP
def genClass(clas, batch_size):
    
    classes = numpy.zeros((batch_size, 10), dtype=numpy.float32)    
    for i in range(batch_size):
        classes[i,clas] = 1.0
    data_tf = tf.convert_to_tensor(classes, numpy.float32)
    
    return data_tf

import numpy as np

##Show a set of even MNIST numbers
def showExamples(Sample):
    cols=5
    rows=int(np.ceil(Sample.shape[0]/cols))
    fig, ax = plt.subplots(nrows=int(np.ceil(Sample.shape[0]/cols)), ncols=cols)
    for i in range(Sample.shape[0]):
        image = (np.reshape(Sample[i,:],(28,28))+1)*-127.5
        plt.subplot(int(np.ceil(Sample.shape[0]/cols)), cols, i+1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        
    for i in range(cols*rows-Sample.shape[0]):
        plt.subplot(int(np.ceil(Sample.shape[0]/cols)), cols, i+Sample.shape[0]+1)
        plt.axis('off')
    plt.show()
    
##Get first N missclasifications of test set
def getNmiss(d_t, N, session):
    #d_t = feed_dict(mnist, False)
    inp_t = {x:d_t[0], y_:d_t[1], keep_prob:1}
    pred = session.run(correct_prediction, feed_dict=inp_t)
    missArg=np.argwhere(pred==False)
    
    return d_t[0][missArg[0:N]]

##Get random sample and label by class
def getRandomSampleByClass(d_t, classNumber):
    labels = np.argmax(d_t[1], axis=1)
    classArg=np.argwhere(labels==classNumber)
    randomInt=np.random.randint(classArg.size-1)
    return d_t[0][classArg[randomInt]], d_t[1][classArg[randomInt]]

#TODO refactor to eliminate code duplication
#Show images RGB
def showExamplesIm(Sample,size,ch):
    cols=5
    rows=int(np.ceil(Sample.shape[0]/cols))
    fig, ax = plt.subplots(nrows=rows, ncols=cols)#not neccesary
    for i in range(Sample.shape[0]):
        image = np.squeeze((np.reshape(Sample[i,...],(size,size,ch))))
        plt.subplot(int(np.ceil(Sample.shape[0]/cols)), cols, i+1)
        if (ch==1): plt.imshow(image, cmap='gray')
        else: plt.imshow(image)
        plt.axis('off')
        
    for i in range(cols*rows-Sample.shape[0]):
        plt.subplot(int(np.ceil(Sample.shape[0]/cols)), cols, i+Sample.shape[0]+1)
        plt.axis('off')
    plt.show()