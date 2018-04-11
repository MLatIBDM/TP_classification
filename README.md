# Machine Learning courses : Image classification using MXNetR framework

Welcome to the Machine Learning practical course !

This practical course is a Tutorial-like on how to build and test programmatically several kind of network topologies,
machine learning technics, in order to produce at the end a very robust machine learning performing on a classification task.

  1. <a name="setupenv"> Setup your work environment </a>
  2. <a name="data">Data :</a>
    1. Load, Import, Normalize and Visualize
    2. Divide your data into Train/Validation/Test set
  3. Network model architecture : Multilayer Perceptron
    1. Train a neural networks (Hyperparameters, Optimizers)
    2. Monitor the results
  4. Data augmentation
  5. Network model architecture: Convolutional Network
    1. Simple LeNet5 network topology
    2. Monitor the internals of a convolutional network
    3. Batch Normalization trick
  6. Transfer Learning


# Setup your working environment [setupenv](#setupenv)
## Import Virtual box prebuilt Ubuntu 16-04 machine and Start Rstudio !

Ask the External hard drive !
To do this practical course, you'll need both to install Virtual Box and the prebuilt MXNet Linux machine.

**Install**<br>

- Install Virtual Box: Go to Directory **TP_Century**, and **Virtual Box**. Choose your executable file according to you operating system and install it.
- Copy the **Virtual Machine** on your computer (you need around 15Go of free space)
- Once **Virtual Box** is installed, launch it.

**Import the machine**<br>
Create a new empty virtual machine:
- Click on the **New** button on top left.
- Enter a name of your machine.
- Choose type: **Linux**
- Choose version: **Ubuntu 64-bit**
- Choose the amount of RAM dedicated to your machine: Set at least **4Go** (ideally 8Go if you have enough RAM free on your computer) and click **Next** button.
- Choose "Use an existing Virtual hard drive" (last option)
- Click on the most left "folder button". Go to the directory into which you have copied the Virtual Machine, and click on the file "MXNET_build.vdi".

Once this is done, you go back on the Machine Menu and just click on **Start** button to launch the virtual machine !

- Log into the machine **login**: mxnet, and **password**: mxnet

Now you are on the desktop, just launch a Terminal by left click on the desktop and choose "Open Terminal".
<br>
You are now ready to start :)
<br>
The machine comes with R, Rstudio and MXNet (CPU version) already installed and properly configured to work with R.
Sublime text editor is also set up in case to edit and code needed scripts for this practical course.

# MXNet Framework

  MXNet is one of the most popular Deep Learning framework. It's available into different programming languages (R, Pyhton, ...) and can handle both working with CPU and GPU.
You'll find more information on the framework API [here](https://mxnet.incubator.apache.org/api/r/index.html)
or visiting their [Github repository](https://github.com/apache/incubator-mxnet)
or by playing with their [tutorials](https://mxnet.incubator.apache.org/tutorials/r/index.html)

# Let's Start: Data ! [data](#data)

**Aim** : Build and Train a classification model which will learn to discriminate a set of bacteria images into four different categories.

### Load, explore, reshape, normalize resample and format the Data for Training !
![Panel](https://github.com/MLatIBDM/TP_classification/blob/master/images/panel.jpg)
<center>Image examples and their corresponding category: </center>
<br>
Check by yourself ! All the images are already well classified into the four categories into <br>
<code>/home/mxnet/TP_classification/DATA/1354.nd2</code> <br>
<code>/home/mxnet/TP_classification/DATA/1354-001.nd2</code> <br>

These repositories contains 1 directory by image category:  <br>
<code>1cell/</code>  <br>
<code>2cell/</code>  <br>
<code>groups/</code>  <br>
<code>rien/</code> <br>

This data will be used for **Training** our model, but also to **Validate** and **Testing** it !

OK now that you know a little bit more the data we will work with, just start RStudio and let's begin explore !

```bash
$ rstudio &
```

If you don't want to work directly into the R console, you can switch to bash console (Linux console) and open an code editor,
in which you'll copy paste the following piece of codes. I've preloaded <em>Sublime Text</em> into the virtual machine. If you want to use it, just type:
```bash
$ subl my_first_network.R &
```

Once into Rstudio, just load the MXNet library by typing the following command into th RStudio console:
```R
library(mxnet)
```

Set up the working directory
```R
source_path = '/home/mxnet/TP/'
setwd(source_path)
````
For the purpose of this practical course, i've already written a bunch of usefull functions to handle data, training ... which will simplify our work today.
**Hint:** To differentiate these functions from MXNet functions, the name of every one start with the prefix <code>mmx.</code>
In case you want to go deeper into this project, you should give a look at the file <code>/home/mxnet/TP/classification_functions.R</code> <br>

Load this usefull functions file:
```R
source(paste(source_path,'classification_functions.R',sep=''))
```
#### Load Data

We want to tell to R where are our images and load it into the R workspace:
```R
path_to_images = list(
                      paste(source_path,'DATA/1354-nd2',sep=''),
                      paste(source_path,'DATA/1354-001-nd2',sep='')
                      )
```
We need to load 3 graphical libraries to properly load and show the data:

```R
library(raster)
library(rgdal)
library(imager)
```
and read the images located into <code>path_to_images</code> repository

```R                      
mydata_orig <- mmx.readDataImages(path_to_images,'*.tif')
```
<code>mydata_orig</code> is a structure (R List) which contains all images and their corresponding label.

#### Explore Data

Let's display for example image number 1 and its corresponding label:

```R
plot (mydata_orig$images[[1]])
mydata_orig$labels[1]
```

You can go further and display the total repartition of the images by labels:
```R
table(mydata_orig$labels)
```
As you noticed, the "label names" are numeric ! MXNet can only handle numeric label, so we need the conversion table: <br>
<code> 0 - 1cell</code> <br>
<code> 1 - 2cell</code> <br>
<code> 2 - Group</code> <br>
<code> 3 - Rien (nothing) </code> <br>


But unfortunately, the data are not ready to be used for **Training** yet !

#### Reshape Data

As you may be noticed the image __Width__ and __Height__ are not uniformized :
```R
width(mydata_orig$images[[1]])
width(mydata_orig$images[[2]])
height(mydata_orig$images[[1]])
height(mydata_orig$images[[2]])
```
The size need to be consistent for every images so as to be processed by our futur classification model, so that we need to reshape all images to a given size.
**Warning:** Choosing an arbitrary too big or too small size will impact significantly our model in both term of prediction accuracy or memory, training time and CPU usage ! So choose wisely if you want to play with this parameter.
A Good compromize is too start with a "rather" small image size like 32 pixels Width anf 32 pixels Height which is rather close to the original image size.

```R
input_image_size = c(32,32,1)
```
The last value is set to "1" because all images have only one colour channel. In the case of RGB, you would have use "3" for example.

Let's do the reshape but before let's make a backup of the original data structure:
```R
mydata <- mydata_orig
mydata <- mmx.reshapeDataImages(mydata,input_image_size)
```
But it's not finished yet :) !

#### Normalize Data

Before doing any **Training** we have now to normalize the pixel intensity values of each images. It exists several way to do it and one which works pretty well is done by something called [Z-Score normalisation](https://en.wikipedia.org/wiki/Standard_score).

```R
mydata <- mmx.normalizeDataImages(mydata)
```

#### Resample Data

One of the first main **issue** of bad accuracy result during **training** is due to the fact that the data are not randomly shuffled.
If you show your data in a non random order, your classifier will start to learn well for this class of data and will converge to their local minimum, but once you show other class it will fail to classify well because you will be stuck in one good local minimum for first data class and not a local minimum achieving a good local minimum for every class.

```R
mydata <- mmx.resampleDataImages(mydata)
```
You can check now that the label and images are randomly sampled.

```R
mydata$labels
```

### Split Data into Training/Validation/Test set

This step is essential to be sure your model is learning well and able to generalize.
We have to split our data into 3 distinct non overlapping set.
 - One which will be use for **Training** our model, we call it **Trainin Set**.

 - One for **Validation**. We will use this set to measure the model prediction accuracy at each steps of the learning.
 **Warning:** This set do not contain any of **Training set** data ! If you a subset of image into **Validation Set** that are currently into the **Training Set**, your model will be greatly biased and you won't be able to know whether you model is able to generalize and/or **overfits** ! Moreover, whithout this set, when later on this practical course ,you'll **fine tune** the network hyperparameters, you won't be able to measure the impact of the generalization capability of your model. We will talk about that later.

 **Important**: Training and Validation set should be __comparable__ in a sense that every one should contains the same proportion of data/class. Otherwise your model will probably fail to achieve good prediction accuracies because it would be overtrained on a subset of data/class.

 - One for **Test**. Once your model has been trained and achieve a good accuracy on validation, we will use this set to measure the predictive capability of your network on a completely new set of data which has not been used for training nor hyperparameters __fine tuning__. This is important because the prediction accuracy on the test set should be very similar on future unlabelled data you'll present to your model.

 Everything is already implemented into the <code> mmx.splitDataImages()</code> functions: split, data/classes equalization accross the different set.

To start we could use this spliting shape: <br>
- Training Set : 60%
- Validation Set : 30%
- Test Set : 10%
<br>

 Let's go !

 ```R
 split_shape = c(60,30,10)
 mydata <- mmx.splitDataImages(mydata,split_shape,equalize=T,epsilon=0.008,maxiter=2000)
 ```
  <code>mydata</code> is a stucture (R list) containing : Training/Validation/Set.
 You can access to the different data set and labels like this:
 ```R
 #Training Set
 mydata$train$images #For Training images
 mydata$train$labels #For Training labels

 #Validation Set
 mydata$valid$images #For Validation images
 mydata$valid$labels #For Validation labels

 #Test Set
 mydata$test$images #For Test images
 mydata$test$labels #For Test labels
  ```

 Check that the repartion data/classes is consistent between the three set:

 ```R
 table(mydata$train$labels)/sum(table(mydata$train$labels))
 table(mydata$valid$labels)/sum(table(mydata$valid$labels))
 table(mydata$test$labels)/sum(table(mydata$test$labels))
 ```
 And lastly we format the data into a readable format for our future neural network:

 ```R
 mydata <- mmx.prepareDataImages(mydata)
 ```

The <code>images</code> are now numeric <code>array</code>.<br>
Check the dimensionality of your training set for example:

```R
  dim(mydata$train$array)
```

You should have something like <code>(32, 32, 1, 404)</code>.<br>
This structure is called a **tensor**. This structure is the only one MXnet can understand to feed a neural network.<br>
In MXNet, a **tensor** is a four dimensional array of size <code>(image_width, image_height, color_depth, number_of_images)</code>.
The tensor shape could be different in case you use another deep learning framework.<br>

Let's display images number 12 for example:

```R
  mydata$train$array[,,,12]
```

You can see that the data could be positive or negative because of the Z-score normalization, but nothing change when you plot it:

```R
 plot(as.cimg(mydata$train$array[,,,12]))
 ```

** Congratulation ** ! Data are now ready to be used for training our model ! :)

 But ... before training the model, we need to design the **Network model architecture** !

 # Network model architecture

 First, we are going to use a classical [**Multi-Layer Perceptron**](https://en.wikipedia.org/wiki/Multilayer_perceptron).
 network architecture for our model.

 ![mlp](https://github.com/MLatIBDM/TP_classification/blob/master/images/mlp.png)

 It's composed of One input Layer, some internal Layer and One output Layer.

 We can build the network hierarchically using MXNet like this:

 ```R
    net <- mx.symbol.Variable("data") #Input Layer
    net <- mx.symbol.FullyConnected(net, name="fc1", num_hidden=128) # First Fully connected (FC) Layer having 128 neurons
    net <- mx.symbol.Activation(net, name="relu1", act_type="relu")  # "Relu" Activation function of the first FC Layer
    net <- mx.symbol.FullyConnected(net, name="fc2", num_hidden=64)  # Second FC Layer having 64 neurons
    net <- mx.symbol.Activation(net, name="relu2", act_type="relu")  # "Relu" Activation function of the second FC Layer
    net <- mx.symbol.FullyConnected(net, name="fc_out", num_hidden=4)   # Output Layer contains 4 neurons: 1 for each image classes  
    net <- mx.symbol.SoftmaxOutput(net, name="sm")                # Output activation "SoftMax"
 ```
Some explanations:<br>
In MXNet we use data type <code>symbol</code> to configure the network. Each layer are "chained" to the previous layer.
We used "relu" as neuron activation function, but we can use some other, here some of them which have their own mathematical properties.
![actf](https://github.com/MLatIBDM/TP_classification/blob/master/images/activation.png)
<br>
<code>FullyConnected</code> refers to a type of neural network layer into which every neuron from the previous layer is connected to every neuron from the current layer.
In our example we have 2 fully connected layers <code>fc1</code> and <code>fc2</code> and 1 output layer <code>fc_out</code> which contains 1 neuron by labels we want to predict.
The last layer <code>SoftmaxOutput</code> will return a probabilistic prediction of our 4 last neurons layer according to their amount of activation. More of softmax function [here](https://en.wikipedia.org/wiki/Softmax_function)

Just check the network architecture to be sure everythig is correct.

```R
graph.viz(net)
```
We will see later on that MXNet provides a wide variety and more complex type of layer usefull for building more complex neural networks.

But from now let's focus on an important step of our model building: the training parameters !

# Training Parameters (Hyperparameters)

In order to achieve good prediction performance, we should take care of the training parameters because they are a really important and critical step.
There is no "magic recipes" i can give you because it's an open research subject nowadays but we are going to highlight some important key to not fail the training.

The two major training parameters to take care of are **learning rate** and **batch size**, but there are other important too.
Here the list of training parameters you can play with:

- **learning_rate :** Define the "velocity" you will travel accross the solution hyperspace during the __gradient descent__ algorithm. A small learning rate means your model will learn and converge very slowly to a local minimum solution. A big learning rate doesn't mean your model will learn quickly but will make bigger "jump" and explor more the solution hyperspace, but it will have difficulties to find a local minimum solution.
So, the learning rate should be choosen wisely to balance "exploration" and "convergeance" to a local minimum.

![lr](https://github.com/MLatIBDM/TP_classification/blob/master/images/lr.png)

- **batch_size :** The gradient descent algorithm use a "trick" to quickly find a local minimum into the solution hyperspace: data are grouped into __mini-batch__ that are used at the same time to calculate the gradient and then train the model. The size of this __mini-batch__ is also important because it will impact greatly the training time, Memory usage and the model performance. Previous studies have shown that the use of __mini-batch__ greatly improves the SGD algorithm.<br>
But, again, be careful : if you use a too large __mini-batch__ size, you will probably explode your computing memory quickly as every example are loaded into RAM. In case it doesn't explode the model should overfit quickly because every batch contains a lot of data and only one gradient is calculated for the mini-batch.<br>
If you use a too small __mini-batch__ size, you won't explode your computing memory, but your gradient will become more instable and you'll probably need to decrease your learning rate to achieve good accuracy.
There is no rules to choose a "good" __mini-batch__ size, you should trial and error a lot before finding a good compromise.
As a general rule, taking 5% or 10% of the total training set size as __mini-batch__ size could be use as a starting point, but once again it's not a general rule.<br>

- **optimizer :** It exists several **gradient descent** algorithm which have their own advantages and counterparts. One of the most used is **SGD (Stochastic Gradient descent)** and its variants: Adagrad, RMSProp, Adam ... [wiki](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)<br>


- **initializer:** Each weight of your network (connecting one neuron to another) is randomly initialized at the begining of the training. MXNet provides several functions to initialize your weights (normal distribution, Xavier distribution, ...). The problem
of weight initialization is an open research problem nowaday.

- **num_round:** The number of epochs you will use to train your model. If you put too much training epochs your model may be would not achieve it's best accuracy. If it's too long, you could be stuck or overfit too much. One of the major technics to prevent overfitting is to stop the learning according to one or several training conditions you have already defined. This is called **early stoping**. You can stop once you achieve a good validation accuracy, or if you model doesn't improve validation accuracy more than one given value for more than X epochs.

- **momentum:** : A momentum it's a kind of "trick" to avoid being stuck into local minimum during solution hyperspace exploration by gradient descent using the previous "velocity" to jump oustide local minima. See [here](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum) for more clear explanation ;)

- **wd (weight decay):** : It regularization trick to add a penalty to the network weights at a given regularization rate (wd). More informations [here](https://en.wikipedia.org/wiki/Convolutional_neural_network#Weight_decay)

>**Play Time !**
>if you want to relax a little bit, you can use this amazing website coded by Google which gives you the opportunity to play with a lot of neural network hyperparameters and architecture and see directly their impact on the training process, a must-see [here](http://playground.tensorflow.org/)


OK Now you know a little bit more about the main training hyperparameters you can play with, let's go !!
<br>
Let's define a first set of hyperparameters:
 ```R
 num_round = 100 # Number of epochs
 batch_size = 60 # the size of the mini-batch
 learning_rate = 0.01
 momentum = 0.9
 wd = 0.00001
```
Last, we need to define the **seed** which will be used for generating the mini_batch order, the network weights ... The seed is a number which will be used by the R random generator.
If you want to keep reproductibility of your results, you should keep the same seed each time.

```R
  mx.set.seed(0)
```

And then proceed to the weight initialization

 ```R
  initializer = mx.init.normal(0.1) #I choose to initialize with this arbitrary value because ... i don't know .. why not ? :)
  ```
We then need to setup one final thing and everything will be ready for the training: <br>
The number of CPU you want to use !<br>
If you don't know how many CPU you have on your computer let <code>nCPU=1</code> in the code below, otherwise feel free to change this value.<br>
```R
nCPU = 1 # the number of CPU assigned to the training
devices <- lapply(1:nCPU,function(i){mx.cpu(i)})
```
**Each __mini-batch__ will be distributed accross the number of CPU you allow for training. So that the number of CPUs should never be smaller than the number of __mini-batch__ .**
<br>

Let's train our network ! :)

## Training

First let's declare a logger which will help us monitor the performance of our model on training and validation set.<br>
We need this because we want to have a visual report on the way our model trains and learn, and we want to be able to detect for examples
whether the model overfits.


```R
logger <- mmx.addLogger(); #Let's declare a logger to log at each epoch the performance of our model
```

We need to indicate to MXNet:
- the network <code>net</code><br>
- the training data <code> mydata$train$array </code> <br>
- the corresponding training labels <code> mydata$train$labels </code> <br>
- the context of execution (CPU) <code> devices </code> <br>
- the number of epochs to train the network <code>num_round</code> <br>
- the initializer policy <code>initializer</code> <br>
- the mini-batch size <code>batch_size</code> <br>
- the learning rate <code> learning_rate</code> <br>
- the <code>momentum</code> and weight decay <code>wd</code>
- the evaluation metric we use to predict the performance of our model. They are several metrics available into MXNet.
For classificiation and softmax as last layer of our network we use an accuracy metric <code>mx.metric.accuracy</code>.
For regression we should use other metric suited for this kind of problems like <code>mx.metric.MSE</code> or <code>mx.metric.RMSE</code>
- the validation data <code>mydata$valid$array</code> and labels <code>mydata$valid$labels</code>
- a callback function called at the end of each epoch to log the accuracy into our logger.

Which gives :

```R

model <- mx.model.FeedForward.create(
                                      net,
                                      X =   mydata$train$array,
                                      y =   mydata$train$labels,
                                      ctx = devices,
                                      num.round = num_round,
                                      initializer = initializer,
                                      array.batch.size = batch_size,
                                      learning.rate = learning_rate,
                                      momentum = momentum,
                                      wd = wd,
                                      eval.metric = mx.metric.accuracy,
                                      eval.data = list(data=mydata$valid$array, label=mydata$valid$labels),
                                      epoch.end.callback = mx.callback.log.train.metric(10,logger)
                                      )
```

At this point you should see your model trying to learn like this: <br>
![train](https://github.com/MLatIBDM/TP_classification/blob/master/images/train.png)

Normally, at the end of the 100 epochs, you should achieve more or less quickly a rather good prediction accuracy on Training set but not so good on validation set !

On my computer, right now as i write this practical course, the network achieves at the end of the 100 epcohs:<br>
- **97% accuracy on Training set**
- **82% accuracy on Validation set**

Check your logger to see more how your model has learnt :

```R
mmx.plotLogger(logger)
```
Here is the mine: <br>
![logger](https://github.com/MLatIBDM/TP_classification/blob/master/images/logger.png)

Congratulation ! We just finished to train our first model together but i'm sorry to tell you that the model we trained seems to be **stuck** and seems to have a little bit **overfits** during the train (around epochs 50). I'm not sure of it's predictive performance.

Let's check on the test data

```R
mmx.evalOn(model,mydata$test)
```
On my computer i achieve : 85% accuracy on test set which is more or less similar to the validation set ... we can do better ! :)

In case here is the complete code:

```R
# --- Load the libraries we'll need
library(mxnet)
library(imager)
libary(rgdal)
library(raster)

# --- Set the source path and functions we need
source_path = '/home/mxnet/TP/'
setwd(source_path)
source(paste(source_path,'classification_functions.R',sep=''))

# --- Data Manager
input_image_size = c(32,32,1)
split_shape = c(60,30,10)

path_to_images = list(
                      paste(source_path,'DATA/1354-nd2',sep=''),
                      paste(source_path,'DATA/1354-001-nd2',sep='')
                      )

  # Read and backup the images
  mydata_orig <- mmx.readDataImages(path_to_images,'*.tif')
  mydata <- mydata_orig

  # Reshape the images
  mydata <- mmx.reshapeDataImages(mydata,input_image_size)

  # Normalize the images
  mydata <- mmx.normalizeDataImages(mydata)

  # Resample the images
  mydata <- mmx.resampleDataImages(my_data)

  # Split into Training/Validation/Test set
  mydata <- mmx.splitDataImages(mydata,split_shape,equalize=T,epsilon=0.008,maxiter=2000)

  # Format the different set to be ready to go into the trainin part
  mydata <- mmx.prepareDataImages(mydata)

# --- Network Manager

  net <- mx.symbol.Variable("data") #Input Layer
  net <- mx.symbol.FullyConnected(net, name="fc1", num_hidden=128) # First Fully connected (FC) Layer having 128 neurons
  net <- mx.symbol.Activation(net, name="relu1", act_type="relu")  # "Relu" Activation function of the first FC Layer
  net <- mx.symbol.FullyConnected(net, name="fc2", num_hidden=64)  # Second FC Layer having 64 neurons
  net <- mx.symbol.Activation(net, name="relu2", act_type="relu")  # "Relu" Activation function of the second FC Layer
  net <- mx.symbol.FullyConnected(net, name="fc_out", num_hidden=4)   # Output Layer contains 4 neurons: 1 for each image classes  
  net <- mx.symbol.SoftmaxOutput(net, name="sm")                # Output activation "SoftMax"

 # --- Training Manager

    # --- Training parameters


      num_round = 100 # Number of epochs
      batch_size = 60 # the size of the mini-batch
      learning_rate = 0.01
      momentum = 0.9
      wd = 0.00001
      mx.set.seed(0)
      initializer = mx.init.normal(0.1)

    # --- Training context
      nCPU = 1
      devices <- lapply(1:nCPU,function(i){mx.cpu(i)})

    # Training
      logger <- mmx.addLogger(); #Let's declare a logger to log at each epoch the performance of our model

      model <- mx.model.FeedForward.create(
                                      net,
                                      X =   mydata$train$array,
                                      y =   mydata$train$labels,
                                      ctx = devices,
                                      num.round = num_round,
                                      initializer = initializer,
                                      array.batch.size = batch_size,
                                      learning.rate = learning_rate,
                                      momentum = momentum,
                                      wd = wd,
                                      eval.metric = mx.metric.accuracy,
                                      eval.data = list(data=mydata$valid$array, label=mydata$valid$labels),
                                      epoch.end.callback = mx.callback.log.train.metric(10,logger)
                                      )

      # Show model performance
      mmx.plotLogger(logger)

      mmx.evalOn(model,mydata$train) # On training set
      mmx.evalOn(model,mydata$valid) # On validation set
      mmx.evalOn(model,mydata$test) # On test set



```
**Let's check the failing cases !**
<br>

It could be interesting to see which images are failing to be well classified.
To do so, we need to first predict, for each images into the **validation** set their corresponding
predicted class probabilities by our **model**.

```R
#On validation set, we do this by one line
predicted_probabilities <- predict(model,mydata$valid$array)
```

For each images into the **validation** set, you can see this 4 class probalities like this:

```R
#Display the 4 class probabilities
predicted_probabilities

# You can check the class probabilities of the first validation image:
predicted_probabilities[,1]
```
Now we need to find, for each of the images, the class which have the maximum probability.

```R
predicted_label = max.col(t(predicted_probabilities))-1
#We put "-1" because the maximum give the position of the class having maximum
#probability starting from 1 to 4. But our labels are ranging from 0 to 3.
```
OK, now we have stored into <code>predicted_label</code> the predicted label for each validation images.<br>
We need now to compare to the expected labels to see which are the images which are failing to be well classified.<br>

First, we can display a [**confusion matrix**](https://en.wikipedia.org/wiki/Confusion_matrix):

```R
table(predicted_label,mydata$valid$labels)
```

We can retrieve the images (by their indices) which fails :

```R
failing_images = which( predicted_label != mydata$valid$labels )
```

and display the failing images to see what's wrong:

```R
#Example the first image which fails
img = mydata$valid$array[,,,failing_images[1]]
plot(as.cimg(img))

#Display the predicted label
predicted_label[failing_images[1]]

#Display the correct label
mydata$valid$labels[failing_images[1]]

```


 **Let's play a bit with learning parameters, Network architecure and Optimizer**
 I let you play a little bit with the learning parameters and network architecture to try to improve your numbers and achieve a better accuracy on both Validation and Test set.<br>
 Try everyting you want: increase/decrease learning size, batch_size, epochs, weight_decay, momentum ...<br>
 Add/remove layers, add/remove more neurons ... <br>
 Change the Trainin/Validation/Test proportions ... <br>
 Increase/decrease the image size ... <br>

You can also play with **Optimizer** and check which one performs the best on your data. <br>
By default, we use **SGD (Stochastic Gradient Descent)** optimizer, and sometimes switching to one optimize to another could improve the validation accuracy.<br>
If you want to use another one: Adam ,RMSProp, Adagrad or Adadelta, just change the Training part like this:
<br>
**<em>Adam</em>**
``` R
#For Adam Optimizer, just delete the <em>momentum<em> parameter and play with learning rate
model <- mx.model.FeedForward.create(
                                net,
                                X =   mydata$train$array,
                                y =   mydata$train$labels,
                                ctx = devices,
                                num.round = num_round,
                                initializer = initializer,
                                array.batch.size = batch_size,
                                learning.rate = learning_rate,
                                wd = wd,
                                eval.metric = mx.metric.accuracy,
                                eval.data = list(data=mydata$valid$array, label=mydata$valid$labels),
                                epoch.end.callback = mx.callback.log.train.metric(10,logger),
                                optimizer="adam"
                                )
```
**<em>Adagrad</em>**
``` R
#For Adam Optimizer, just delete the <em>momentum<em> parameter and play with learning rate
model <- mx.model.FeedForward.create(
                                net,
                                X =   mydata$train$array,
                                y =   mydata$train$labels,
                                ctx = devices,
                                num.round = num_round,
                                initializer = initializer,
                                array.batch.size = batch_size,
                                learning.rate = learning_rate,
                                wd = wd,
                                eval.metric = mx.metric.accuracy,
                                eval.data = list(data=mydata$valid$array, label=mydata$valid$labels),
                                epoch.end.callback = mx.callback.log.train.metric(10,logger),
                                optimizer="adagrad"
                                )
```
**<em>Adadelta</em>**
``` R
#For Adam Optimizer, just delete the <em>momentum<em> parameter and learning rate.
# You need to add two new hyperperameters: "rho" and "epsilon"
model <- mx.model.FeedForward.create(
                                net,
                                X =   mydata$train$array,
                                y =   mydata$train$labels,
                                ctx = devices,
                                num.round = num_round,
                                initializer = initializer,
                                array.batch.size = batch_size,
                                wd = wd,
                                eval.metric = mx.metric.accuracy,
                                eval.data = list(data=mydata$valid$array, label=mydata$valid$labels),
                                epoch.end.callback = mx.callback.log.train.metric(10,logger),
                                optimizer="adadelta",
                                rho="0.9",
                                epsilon=1e-04
                                )
```
**<em>RMSProp</em>**
```R
#For RMSProp Optimizer, just delete the <em>momentum<em> parameter, and play with learning rate
model <- mx.model.FeedForward.create(
                                net,
                                X =   mydata$train$array,
                                y =   mydata$train$labels,
                                ctx = devices,
                                num.round = num_round,
                                initializer = initializer,
                                array.batch.size = batch_size,
                                learning.rate = learning_rate,
                                wd = wd,
                                eval.metric = mx.metric.accuracy,
                                eval.data = list(data=mydata$valid$array, label=mydata$valid$labels),
                                epoch.end.callback = mx.callback.log.train.metric(10,logger),
                                optimizer="rmsprop"
                                )
```
Hint: RMSProp works well with really small learning rates (in comparison to SGD)

Have fun ! You can crash the machine, it's a virtual one :) !


# Data Augmentation


After tuning the network, the parameters, the sets without major accuracy improvements ... One of the main trick to improve prediction accuracy is to artificially augment the data size to show the model more training data. This is called Data Augmentation. How to augment the data without adding bias to the final prediction ? By taking images from the training et and applying some subtle modifications like rotation, flip, noising, ... See [here](https://github.com/aleju/imgaug) for a almost complete list of augentation you can apply to your images without adding any bias to the training process.

Let's try Data Augmentation on our dataset:

<b>Important</b> : As we don't want to bias the results on Validation, we must do the data augmentation ONLY on the training set

First we need to load a new library which will help us to play with the data
```R
  library(abind)
```


Then, we keep a copy of our training set into <code>base</code>

```R
	base = mydata$train;
```

For the purpose of this practical course we are just doing 3 type of data augmentation:
<li> Rotation </li>

```R
	base_rotated <- mmx.rotateDataImages(base,c(90,-90,270,-270,-45,45))
```
The second part of the arguments is a vector containing the desired rotation angle, i chose this ones, but feel free to test
whatever angle you want !


<li> Dimension fliping </li>

```R
	base_flipped <- mmx.flipDataImages(base)
```

<li> Noising </li>

```R
	base_noised <- mmx.noiseDataImages(base,m=0,sd=0.10)
```
The second and third argument are the parameters for the Gaussian noise distribution, in this case i choose mean=0 and standard deviation=0.10. Again feel free to play with it.


Finally we merge all the data augmentations into one unique dataset that will serve as "Augmented" Training set.
We set the parameter <code>resample</code> to True, because we want to shuffle up every images into our augmented training dataset.

```R
	mydata$train <- mmx.mergeDataImages(mydata$train,list(base_rotated,base_noised,base_flipped),resample=T)
```

That's it !
Now let's redo the training one more time using our Augmented training Set:

```R
 # --- Training Manager

    # --- Training parameters
      num_round = 100 # Number of epochs
      batch_size = 60 # the size of the mini-batch
      learning_rate = 0.01
      momentum = 0.9
      wd = 0.00001
      mx.set.seed(0)
      initializer = mx.init.normal(0.1)

    # --- Training context
      nCPU=1
      devices <- lapply(1:nCPU,function(i){mx.cpu(i)})


    # Training
      logger <- mmx.addLogger(); #Let's declare a logger to log at each epoch the performance of our model

      model <- mx.model.FeedForward.create(
                                      net,
                                      X =   mydata$train$array,
                                      y =   mydata$train$labels,
                                      ctx = devices,
                                      num.round = num_round,
                                      initializer = initializer,
                                      array.batch.size = batch_size,
                                      learning.rate = learning_rate,
                                      momentum = momentum,
                                      wd = wd,
                                      eval.metric = mx.metric.accuracy,
                                      eval.data = list(data=mydata$valid$array, label=mydata$valid$labels),
                                      epoch.end.callback = mx.callback.log.train.metric(10,logger)
                                      )

      # Show model performance
      mmx.plotLogger(logger)

      mmx.evalOn(model,mydata$train) # On training set
      mmx.evalOn(model,mydata$valid) # On validation set
      mmx.evalOn(model,mydata$test) # On test set
```

What do you notice ?

On my computer, I achieved, after 100 epoch an accuracy of **98% on training set** and around **90% on validation set** ...
It's far better, but we can do much better because we chose a network architecture not very well suited to handle images, even if we applied
data augmentations.<br>
Multilayer perceptron are weak when you have to classify an image, because it uses every pixels of the images to feed the network, and become totally biased by the scale/translation variations.<br><br>
We need a network topology which can achieve better prediction by taking into account objects, despite their scale, orientation, translation, position variations.

# Convolutional Neural Network

Yes ! Convolutional could help us building a more accurate prediction program for our classification task.

![dcn](https://github.com/MLatIBDM/TP_classification/blob/master/images/dcn.png)

To do so, we need to setup a new network structure. I chose to let you play with the <em>original</em> version of Convolutional architecture called **LeNet5** published in 1998 by Yann Lecun & al. ([PDF](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf))
<br>
<br>
Here is the network architecture : <br>
![lenet5](https://github.com/MLatIBDM/TP_classification/blob/master/images/lenet5.png)
<br>

Ok let's start building it !

```R
First define the input:
# input
net_lenet5 <- mx.symbol.Variable('data')
```
Then the **first** convolution layer [wiki](https://en.wikipedia.org/wiki/Convolutional_neural_network#Convolutional_layer)
```R
# first conv
net_lenet5 <- mx.symbol.Convolution(data=net_lenet5, kernel=c(5,5), num_filter=6)
net_lenet5 <- mx.symbol.Activation(data=net_lenet5, act_type='relu')
net_lenet5 <- mx.symbol.Pooling(data=net_lenet5, pool_type="max",kernel=c(2,2), stride=c(2,2))
```
Some explanations:
We defined the convolution layer using <code>mx.symbol.Convolution</code>. We then defined the convolution kernel using <code>kernel</code> parameter and set it to a square windows of size <code>5x5</code>. This kernel will represent the number of
neurons (5x5=25 neurons) which will look around the images.<code>num_filter</code> represents the number of different convolution kernel we'll use for this first layer. In our case we put 6 different convolution kernel. The more you use, the more different details of the image you'll capture.<br>
We put then put an <code>mx.symbol.Activation</code> layer, using the <code>relu</code> activation type.<br>
The last line of the first convolutional network deals with the way we are reducing the image size.<br>
I remind you that the goal of a convolution layer is to reduce and compress the information. That's why we use a <code>mx.symbol.Pooling</code> layer at the end [wiki](https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer). We define the way we want to apply the pooling : <code>pool_type</code> as <code>max</code> into a sliding windows defined by the <code>kernel</code> arguments of size: <code>2x2</code>. <br>
The last arguments define the <code>stride</code>, it define how many pixels you are "jumping" during the sliding process. On our case we chose a stride of <code>2x2</code>. We chose the same size for the kernel and the stride which means that we want the kernel sliding two pixels by two.
<br>
<br>
**To sum up**: This first layer will be able to learn and detect at least 6 differents small patches of size 5x5 of your images. Will operate a Relu activation on each neurons to retrieve an image of size 28x28 (because of the convolution kernel of size 5 : 32-5+1) and then apply a pooling of size 2x2 with stride 2x2 decreasing your image to a size of 14x14 (28/2).
Moreover, as we defined 6 filters, we will have at the end of this layer 6 images (one by filters)
<br><br>
Let's jump to the second layer !
<br>

```R
# second conv
net_lenet5 <- mx.symbol.Convolution(data=net_lenet5, kernel=c(5,5), num_filter=16)
net_lenet5 <- mx.symbol.Activation(data=net_lenet5, act_type='relu')
net_lenet5 <- mx.symbol.Pooling(data=net_lenet5, pool_type="max",kernel=c(2,2), stride=c(2,2))
```
Some explanations:<br>
Same schema as before, we chose the same kernel size (5x5) but we increased the number of convolution filters to 16.
So now, for each of our 6 previous images calculated at the first layer, we have 16 new images computed by this new convolution kernel.
We should normally have now 6x16 images = 96 images in total.
After doing convolution we decreases the 96 images sizes to 10x10, and after doing max pooling : the images size are 5x5.
<br>

**Warning & advises**:
For this particular network architecture that's it for the convolutional layer, but you can find very deep architectures which can contains more than 10 convolutional layers !
Adding more and more layer is very memory and training time demanding. Some of the state of the art architecture for Image classification are really deep and are almost
impossible to run on a standard personal computer.<br>
In this practical course, we work only with CPU and not too much amount of RAM are needed to make this architecture work well. But if you choose to work with deeper architecture you should switch first to use GPU
with a big amount of "Graphical RAM".
<br>
Go back to our network :
```R
# flatten layer
net_lenet5 <- mx.symbol.Flatten(data=net_lenet5)
```
This layer is a kind of magical layer ! He's in charge of transforming our images into vector. In other words it will transform our final 5x5 96 images, into a vector of size 2400 (5x5x96).
At this point, we achieved the "compression" of our images into a set of 2400 features because this vector is the compressed version of our original image !
<br>
We need now to pass this vector to the "brain".
<br>
```R
# fully connected layer 1
net_lenet5 <- mx.symbol.FullyConnected(data=net_lenet5, num_hidden=120)
net_lenet5 <- mx.symbol.Activation(data=net_lenet5, act_type='relu')
# fully connected layer 2
net_lenet5 <- mx.symbol.FullyConnected(data=net_lenet5, num_hidden=84)
net_lenet5 <- mx.symbol.Activation(data=net_lenet5, act_type='relu')
  # fully connected layer 3 (output)
net_lenet5 <- mx.symbol.FullyConnected(data=net_lenet5, num_hidden=4)
```
The goal of this last layer will be to process our feature vector to generate our class prediction. We have 2 layers, the first one with 120 neurons and the second one having 84.
The last fully connected layer contains as many classes as we want to classify: 4.
<br>
Finally, we add the loss layer using the softmax for accuracy prediction of our 4 classes.
<br>
```R
# loss layer
net_lenet5 <- mx.symbol.SoftmaxOutput(data=net_lenet5,name = 'softmax')
```

Check the new architecture :
```R
  graph.viz(net_lenet5)
```

If everyting is ok, we are ready to test this new network toplogy :)

```R
# --- Training Manager

   # --- Training parameters
     num_round = 100 # Number of epochs
     batch_size = 60 # the size of the mini-batch
     learning_rate = 0.01
     momentum = 0.9
     wd = 0.00001
     mx.set.seed(0)
     initializer = mx.init.normal(0.1)

   # --- Training context
     nCPU=1
     devices <- lapply(1:nCPU,function(i){mx.cpu(i)})


   # Training
     logger <- mmx.addLogger(); #Let's declare a logger to log at each epoch the performance of our model

     model <- mx.model.FeedForward.create(
                                     net_lenet5,
                                     X =   mydata$train$array,
                                     y =   mydata$train$labels,
                                     ctx = devices,
                                     num.round = num_round,
                                     initializer = initializer,
                                     array.batch.size = batch_size,
                                     learning.rate = learning_rate,
                                     momentum = momentum,
                                     wd = wd,
                                     eval.metric = mx.metric.accuracy,
                                     eval.data = list(data=mydata$valid$array, label=mydata$valid$labels),
                                     epoch.end.callback = mx.callback.log.train.metric(10,logger)
                                     )

     # Show model performance
     mmx.plotLogger(logger)

     mmx.evalOn(model,mydata$train) # On training set
     mmx.evalOn(model,mydata$valid) # On validation set
     mmx.evalOn(model,mydata$test) # On test set


```

What do you see ?

On my computer, i can achieve **97% on training set** and between **91/93% on validation set** at best epoch iteration.
<br>
So by just switching from an architecture to another, we increased the accuracy in a postive way !
<br>

**And what if we inspect the convolutional layers internals ?** <br>

First let's explore the internals of our model.<br>
All the model weights are stored into the model$arg.params structure:

```R
#Retrieve the names of the layers
names(model$arg.params)

#Choose the first convolutional layer called: convolution0_weight, and convert it
#into an array.
first_conv = as.array(model$arg.params$convolution0_weight)

#Let's plot all the shape of the first convolutional layer
par(mfrow=c(2,3))
for (i in 1:dim(first_conv)[4]){
  plot(as.cimg(first_conv[,,,i]))
}


```
This is what the convolution kernels of the first convolution layer looks like. <br>
Now why don't checking what the network really see ? <br>
To do so, we are going to do the **convolution product** between an image and all the first convolution layer kernels <br>

```R

#Let's take for example the second image of our training dataset and make the convolution product
c1 = convolve ( as.cimg(mydata$train$array[,,,2]) , first_conv )

#Convert the convolution result into an array
array_c1 = as.array(c1)

#Let's plot it !
par(mfrow=c(2,3))
for ( i in 1:dim(array_c1)[4]){
  plot(as.cimg(array_c1[,,,i]))
}

```
This is what your network is really seeing ! <br>

Then, let's simulate the <code>Relu</code> activation function:

```R
#Check the negative pixels and put them to 0 value (Relu activation)
array_c1[which(array_c1<0)]=0

#plot the result
par(mfrow=c(2,3))
for ( i in 1:dim(array_c1)[4]){
  plot(as.cimg(array_c1[,,,i]))
}

```

You notice that each convolutional kernel has got a specific role into the detection of some part of your image: some are detecting edges, some the bacteria, some the background ...
<br>
This is the goal of your convolution kernels: detect some specific parts, which will be further combined together to produce your classification prediction.

<br>
<br>

OK, let's go back to our aim: increase accuracy ! <br>
We probably can achieve a slightly better score using a trick called Batch Normalization !

**Batch Normalization**
<br>
Batch Normalization is a technique which will help you Normalize your batch ... really ... lol ... but how ?<br>
We know that normalization (Z-score normalization : zero-mean and unit variance) is often used when we pre-process images to make data comparable.
But as the data flows through your network, the weights adjusts those values and sometime "denormalize" the data.
If you do this normalization operation into each mini-batch, once you train your network, you avoid this problem, and this very powerfull method is called <em>Batch Normalization</em>.
<br>
Let's create a new network, based on the convolutional network we've just implemented, with Batch Normalization layers:<br>

```R
# input
	net_lenet5_BN <- mx.symbol.Variable('data')
	# first conv + Batch Normalization
	net_lenet5_BN <- mx.symbol.Convolution(data=net_lenet5_BN, kernel=c(5,5), num_filter=6)
	net_lenet5_BN <- mx.symbol.BatchNorm(data=net_lenet5_BN)
  net_lenet5_BN <- mx.symbol.Activation(data=net_lenet5_BN, act_type='relu')
	net_lenet5_BN <- mx.symbol.Pooling(data=net_lenet5_BN, pool_type="max",kernel=c(2,2), stride=c(2,2))

	# second conv + Batch Normalization
	net_lenet5_BN <- mx.symbol.Convolution(data=net_lenet5_BN, kernel=c(5,5), num_filter=16)
	net_lenet5_BN <- mx.symbol.BatchNorm(data=net_lenet5_BN)
	net_lenet5_BN <- mx.symbol.Activation(data=net_lenet5_BN, act_type='relu')
	net_lenet5_BN <- mx.symbol.Pooling(data=net_lenet5_BN, pool_type="max",kernel=c(2,2), stride=c(2,2))
	# flatten layer
	net_lenet5_BN <- mx.symbol.Flatten(data=net_lenet5_BN)
	# fully connected layer 1 + Batch_Normalization
	net_lenet5_BN <- mx.symbol.FullyConnected(data=net_lenet5_BN, num_hidden=120)
	net_lenet5_BN <- mx.symbol.BatchNorm(data=net_lenet5_BN)
	net_lenet5_BN <- mx.symbol.Activation(data=net_lenet5_BN, act_type='relu')
	# fully connected layer 2 + Batch_Normalization
	net_lenet5_BN <- mx.symbol.FullyConnected(data=net_lenet5_BN, num_hidden=84)
	net_lenet5_BN <- mx.symbol.BatchNorm(data=net_lenet5_BN)
	net_lenet5_BN <- mx.symbol.Activation(data=net_lenet5_BN, act_type='relu')
		# fully connected layer 3 (output)
	net_lenet5_BN <- mx.symbol.FullyConnected(data=net_lenet5_BN, num_hidden=4)
	# loss layer
	net_lenet5_BN <- mx.symbol.SoftmaxOutput(data=net_lenet5_BN,name = 'softmax')
```
and restart the training :  <br>

```R
# --- Training Manager

   # --- Training parameters
     num_round = 100 # Number of epochs
     batch_size = 60 # the size of the mini-batch
     learning_rate = 0.01
     momentum = 0.9
     wd = 0.00001
     mx.set.seed(0)
     initializer = mx.init.normal(0.1)

   # --- Training context
     nCPU=1
     devices <- lapply(1:nCPU,function(i){mx.cpu(i)})

   # Training
     logger <- mmx.addLogger(); #Let's declare a logger to log at each epoch the performance of our model

     model <- mx.model.FeedForward.create(
                                     net_lenet5_BN,
                                     X =   mydata$train$array,
                                     y =   mydata$train$labels,
                                     ctx = devices,
                                     num.round = num_round,
                                     initializer = initializer,
                                     array.batch.size = batch_size,
                                     learning.rate = learning_rate,
                                     momentum = momentum,
                                     wd = wd,
                                     eval.metric = mx.metric.accuracy,
                                     eval.data = list(data=mydata$valid$array, label=mydata$valid$labels),
                                     epoch.end.callback = mx.callback.log.train.metric(10,logger)
                                     )

     # Show model performance
     mmx.plotLogger(logger)

     mmx.evalOn(model,mydata$train) # On training set
     mmx.evalOn(model,mydata$valid) # On validation set
     mmx.evalOn(model,mydata$test) # On test set


```

Do you see an improvment ?

From my part, yes i achieved a slightly better accuracy: **99% on training set** and aournd **94/95% on validation set** : Great !

I let you play a bit with hyperparameters, network architecture, optimize ... and why not trying to get better accuracy !


# Transfer Learning

How to convert a <em>"Cat & Dog"</em> model into a really powerful model able to classify our bacteria accurately ? **Transfer Learning** <br>
<br>
What is transfer Learning ?
![tl](https://github.com/MLatIBDM/TP_classification/blob/master/images/tl.png)
Transfer Learning allow us to convert an already trained model for a specific classification task into another model dedicated to another classification task. <br>
It exists several already pretrained model on plenty of different classification task, and for the purpose of this practical course, we will use **GoogleLenet - Inception BN** model.<br>
This model, created by **Google** has been train on millions of images, depicting cars, plane, fruits, animals ... from **ImageNet** [database](http://www.image-net.org).
GoogleLenet is a rather complex model which gather a lot of convolution layers (actually 22 convolutional layer), and use some topologies trick to improve the classification score on the **Imagenet** dataset. See [here](https://arxiv.org/abs/1409.4842) the original paper.
<br>
**Bad news :** As this architecture is absolutely huge in terms of memory and time consuming, it's impossible to train on such model from scratch. <br>
**Good news:** Other guys have already trained it on **ImageNet** and we just have to transfer it to our problem by only retraining the <em>"brain"</em> part using a **Multi-layer Perceptron**, like we did at the begining of this practical course !
<br>
OK ! But we need a little bit of work to have a such a result, so let's start !
<br>
### DATA regeneration
We have to regenerate our data to have a clean dataset:
```R
mydata_orig <- mmx.readDataImages(path_to_images,'*.tif')
mydata <- mydata_orig
```

Once it's done, and as **GoogleLenet** model works in RGB, we have to convert our images into RGB, and normalize + resample the images:

```R
input_image_size = c(32,32,3)
mydata <- mmx.reshapeDataImages(mydata,input_image_size)
mydata <- mmx.normalizeDataImages(mydata)
mydata <- mmx.resampleDataImages(mydata)
```
Make the train/validation/test split like before:

```R
split_shape = c(60,30,10)
mydata <- mmx.splitDataImages(mydata,split_shape,equalize=T,epsilon=0.008,maxiter=2000)
```
Ok now, we have clean dataset, we need to work now a bit on GoogleLenet model.

### GoogleLenet model

First we have to load the pretrained model, I've already download into your virtual machine.
We have 2 files:
- <code>Inception_BN-0039.params</code> which contains the pretrained networks weights
- <code>Inception_BN-symbol.json</code> which contains the architecture of the network (its topology)
<br>
Let's do this:

```R
googlelenet_model = mx.model.load(paste(source_path,"inception_model/Inception_BN",sep=''), iteration=39)
```

Now we need to empyt the last fully connected layer of this model (<em>"brain"</em> part) and to keep the convolutionnal layers (<em>"eyes"</em> part). <br>
**The goal is to have a modified GoogleLenet model which takes as input an image and outputs its corresponding features (feature vector)**<br>
This is what we do by doing this:
```R
#We extract the weights of the GoogleLenet network
internals = googlelenet_model$symbol$get.internals()

#We copy the weights from the begining to the last convolutional layers which contains the features
fea_symbol = internals[[match("global_pool_output", internals$outputs)]]

#We empty the brain parts
googlelenet_model$arg.params$fc_bias <- NULL
googlelenet_model$arg.params$fc_weight <- NULL

#And we create a new model like we wanted
googlelenet_model_features <- list(symbol = fea_symbol,
              arg.params = googlelenet_model$arg.params,
              aux.params = googlelenet_model$aux.params)

class(googlelenet_model_features) <- "MXFeedForwardModel"
```
### Features

Ok now we have both our data and a shortened GoogleLeNet model, we want to retrieve our features by putting our images into the shortened model.
I've written a dedicated function to do so, feel free to check it into <code>classification_functions.R</code> script.

```R

#Calculate the feature vectors for Training dataset
train_feature_vectors <- mmx.calculateFeatures(mydata$train,googlelenet_model_features)
valid_feature_vectors <- mmx.calculateFeatures(mydata$valid,googlelenet_model_features)
test_feature_vectors <- mmx.calculateFeatures(mydata$test,googlelenet_model_features)
```

And create the train/validation/test structure to be use for the trains
```R
train_features <- mmx.prepareDataFeatures(train_feature_vectors)
valid_features <- mmx.prepareDataFeatures(valid_feature_vectors)
test_features <- mmx.prepareDataFeatures(test_feature_vectors)
```

### Training
We are going to use a basic **Multilayer perceptron** for the brain part.<br>
Let's define it:

```R
   googlelenet_model_brain <- mx.symbol.Variable("data") #Input Layer
   googlelenet_model_brain <- mx.symbol.FullyConnected(googlelenet_model_brain, name="fc1", num_hidden=128) # First Fully connected (FC) Layer having 128 neurons
   googlelenet_model_brain <- mx.symbol.Activation(googlelenet_model_brain, name="relu1", act_type="relu")  # "Relu" Activation function of the first FC Layer
   googlelenet_model_brain <- mx.symbol.FullyConnected(googlelenet_model_brain, name="fc2", num_hidden=64)  # Second FC Layer having 64 neurons
   googlelenet_model_brain <- mx.symbol.Activation(googlelenet_model_brain, name="relu2", act_type="relu")  # "Relu" Activation function of the second FC Layer
   googlelenet_model_brain <- mx.symbol.FullyConnected(googlelenet_model_brain, name="fc_out", num_hidden=4)   # Output Layer contains 4 neurons: 1 for each image classes  
   googlelenet_model_brain <- mx.symbol.SoftmaxOutput(googlelenet_model_brain, name="sm")                # Output activation "SoftMax"
```

and now let's redefine the hyperparameters:

```R
mx.set.seed(0)
num_round = 100 # Number of epochs
batch_size = 60 # the size of the mini-batch
learning_rate = 0.01
momentum = 0.9
wd = 0.00001
initializer = mx.init.normal(0.1)

#And the context
nCPU = 1 # the number of CPU assigned to the training
devices <- lapply(1:nCPU,function(i){mx.cpu(i)})


```

OK ! Now we are ready to train and see whether we improve the accuracy or not .... <br>

```R
#Add a logger
logger <- mmx.addLogger(); #Let's declare a logger to log at each epoch the performance of our model

#Train !
model <- mx.model.FeedForward.create(
                                      googlelenet_model_brain,
                                      X =   train_features$array,
                                      y =   train_features$labels,
                                      ctx = devices,
                                      num.round = num_round,
                                      initializer = initializer,
                                      array.batch.size = batch_size,
                                      learning.rate = learning_rate,
                                      momentum = momentum,
                                      wd = wd,
                                      eval.metric = mx.metric.accuracy,
                                      eval.data = list(data=valid_features$array, label=valid_features$labels),
                                      epoch.end.callback = mx.callback.log.train.metric(10,logger)
                                      )


```

As you can see, the accuracy doesn't seem to improve greatly at first try. According to me, the original image size doesn't fit well with this kind of network because
we needed to increase their size too much. By doing this, we needed some image interpolation and we probably lost much useful information and introduced too much noise into the images.<br>
It was just a proof of concept of how **transfer learning** works and how to use it for our purpose :)
