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

### Data
![Panel](https://github.com/MLatIBDM/TP_classification/blob/master/images/panel.jpg)
<center>Image examples and their corresponding category: </center>
<br>
Check by yourself ! All the images are already well classified into the four categories into 
<code>/home/mxnet/TP_classification/DATA/1354.nd2</code> and <br>
<code>/home/mxnet/TP_classification/DATA/1354-001.nd2</code> <br>
 
These repositories contains 1 directory by image category:  <br>
<code>/home/mxnet/TP_classification/DATA/1354.nd2/1cell/</code>  <br>
<code>/home/mxnet/TP_classification/DATA/1354.nd2/2cell/</code>  <br>
<code>/home/mxnet/TP_classification/DATA/1354.nd2/groups/</code>  <br>
<code>/home/mxnet/TP_classification/DATA/1354.nd2/rien/</code> <br>

 <br>








## Play with main.R
```bash
$ rstudio &
```
Once Rstudio is loaded : check MXNet is ready to work by loading it into the R Wokspace
```R
> library(mxnet)
```



