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

<h1> Step 1: Import Virtual box prebuilt machine </h1>
For those who don't have already downloaded Virtual box, please download :<br>
<li> <a href="https://www.virtualbox.org/"> Virtual box </a></li>
<li> Activate Virtualization into the BIOS (in case it's not lready done)</li>
<li> Import the virtual machine from <a href="_blank">here</a> </li>

Once it's done, you must start the machine (login::password / mxnet::mxnet)

<h1> Step 2: Start RStudio from Docker instance into the machine </h1>

Open a terminal (Ctrl+Shift+t) and type: <br>

Go to TP directory: <br>
<code> cd ~/TP/ </code><br>

Start the docker instance:<br>
<code> sudo docker run -d -p 8787:8787 -v $(pwd):/home/TP/ f0nzie/rstudio-mxnet </code><br>

Start Rstudio client:<br>
<code> sudo firefox localhost:8787& </code><br>

Use login::password / rstudio::rstudio, to login to the Rstudio server <br>

