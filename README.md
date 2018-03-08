# Machine Learning courses : Image classification

Welcome to the Deep Learning practical course !

<b>Goals:</b><br>
- Be familiar with basic operation using MXNet-R framework
- Play with different deep learning model architectures
- Be aware of important steps involved during training process
- Quantify the models accuracy

<b> Toy problem: </b>
- During this practical course you will build a model which will learn to classify bacteria images into distinct classes.

<b> Input data </b>
- A set of bacteria images

<b> Output data </b>
- 4 Classes: 1-cell, 2-cell, more-than-2cell, not-a-bacterua

<h1> Setup your working environment: Import Virtual box prebuilt Ubuntu 16-04 machine and Start Rstudio !</h1>
For those who don't have already downloaded Virtual box, please download :<br>
<li> <a href="https://www.virtualbox.org/"> Virtual box </a></li>
<li> Activate Virtualization into the BIOS (in case it's not already done)</li>
<li> Import the virtual machine from <a href="_blank">here</a> </li>
<li Start the machine (login::password / mxnet::mxnet)</li>

The machine comes with R, Rstudio and MXNet already installed and properly configured to work with R. 
Sublime text editor is also set up to edit and code the needed scripts for this practical course.

OK ! Now that you are logged into the virtual machine, you are ready to Go ! :)

Open a terminal (Ctrl+Shift+t) and type: <br>

```bash
$ rstudio &
```
Once Rstudio is loaded : check MXNet is ready to work by loading it into the R Wokspace
```R
> library(mxnet)
```

<h1> MXNet Framework </h1>

MXNet is one of the most popular Deep Learning framework. It's available into different programming languages (R, Pyhton, ...) and can handle both working with CPU and GPU. 





