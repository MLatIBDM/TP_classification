# Machine Learning courses : Image classification using MXNet framework for R 

Welcome to the Machine Learning practical course !

# Setup your working environment
## Import Virtual box prebuilt Ubuntu 16-04 machine and Start Rstudio !

For those who don't have already downloaded Virtual box, please download :<br>
- [Virtual box]("https://www.virtualbox.org/")
- Activate Virtualization into the BIOS (in case it's not already done)
- Download the virtual machine from [here]()
- Import the virtual machine into Virtual box (RAM ? Disk space ? ...)
- Start the machine (login::password / mxnet::mxnet)

The machine comes with R, Rstudio and MXNet already installed and properly configured to work with R. 
Sublime text editor is also set up in case to edit and code needed scripts for this practical course.

# MXNet Framework 

  MXNet is one of the most popular Deep Learning framework. It's available into different programming languages (R, Pyhton, ...) and can handle both working with CPU and GPU. 
You'll find more information on the framework API [here](https://mxnet.incubator.apache.org/api/r/index.html)
or visiting their [Github repository](https://github.com/apache/incubator-mxnet)
or by playing with their [tutorials](https://mxnet.incubator.apache.org/tutorials/r/index.html)

# Let's Start !

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
                      '/home/mxnet/TP/DATA/1354-nd2',
                      '/home/mxnet/TP/DATA/1354-001-nd2'
                      )
mydata_orig <- mmx.readDataImages(path_to_images,'*.tif')
```
<code>mydata_orig</code> is a structure (R List) which contains all images and their corresponding label.

#### Explore Data

If you want to display one given image, you need to load the **imager** library:

```R
library(imager)
````
and then display for example the image number 1 and its corresponding label:

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
A Good compromize is too start with a "rather" small image size like 32 pixels Width anf 32 pixels Height.

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
mydata <- mmx.resampleDataImages(my_data)
```
You can check now that the label and images are randomly sampled.

```R
mydata$orig$labels
```

### Split Data into Training/Validation/Test set

This step is essential to be sure your model is learning well and able to generalize. 
We have to split our data into 3 distinct non overlapping set.
 - One which will be use for **Training** our model, we call it **Trainin Set**.
 - One for **Validation**. We will use this set to measure the model prediction accuracy at each steps of the learning.
 **Warning:** This set do not contain any of **Training set** data ! If you a subset of image into **Validation Set** that are currently into the **Training Set**, your model will be greatly biased and you won't be able to know whether you model is able to generalize and/or **overfits** ! Moreover, whithout this set, when later on this practical course ,you'll **fine tune** the network hyperparameters, you won't be able to measure the impact of the generalization capability of your model. We will talk about that later.
 - One for **Test**. Once your model has been trained and achieve a good accuracy on validation, we will use this set to measure the predictive capability of your network on a completely new set of data which has not been used for training nor hyperparameters __fine tuning__. This is important because the prediction accuracy on the test set should be very similar on future unlabelled data you'll present to your model.
 
 **Important**: All three set should be __comparable__ in a sense that every one should contains the same proportion of data/class.
 Otherwise your model will probably fail to achieve good prediction accuracies.










