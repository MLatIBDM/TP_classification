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

<h1> Setup your working environment: Import Virtual box prebuilt machine and Start Rstudio !</h1>
For those who don't have already downloaded Virtual box, please download :<br>
<li> <a href="https://www.virtualbox.org/"> Virtual box </a></li>
<li> Activate Virtualization into the BIOS (in case it's not lready done)</li>
<li> Import the virtual machine from <a href="_blank">here</a> </li>
<li Start the machine (login::password / mxnet::mxnet)</li>

Now, you are logged into the virtual machine and ready to start the work :)

Open a terminal (Ctrl+Shift+t) and type: <br>

```bash
$ rstudio &
```
Once Rstudio is loaded : check MXNet is ready to work by loading it into the R Wokspace
```R
> library(mxnet)
```

<h1> Exercice 1 : </h1>
