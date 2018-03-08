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

OK ! Now that you are logged into the virtual machine, you are ready to Go ! :)

Open a terminal (Ctrl+Shift+t) and type: <br>

```bash
$ rstudio &
```
Once Rstudio is loaded : check MXNet is ready to work by loading it into the R Wokspace
```R
> library(mxnet)
```

# MXNet Framework 

  MXNet is one of the most popular Deep Learning framework. It's available into different programming languages (R, Pyhton, ...) and can handle both working with CPU and GPU. 
You'll find more information on the framework API [here](https://mxnet.incubator.apache.org/api/r/index.html)
or visiting their [Github repository](https://github.com/apache/incubator-mxnet)
or by playing with their [tutorials](https://mxnet.incubator.apache.org/tutorials/r/index.html)

# The problem
## Build and Train a classification model able to discriminate a set of bacteria images into four categories

Inline-style: 
![Panel](https://github.com/MLatIBDM/TP_classification/blob/master/images/panel.jpg)




