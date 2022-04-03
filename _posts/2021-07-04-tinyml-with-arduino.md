---
id: 639
title: 'TinyML with Arduino'
date: '2021-07-04T15:40:17+02:00'
author: qooba
layout: post
guid: 'https://qooba.net/?p=639'
permalink: /2021/07/04/tinyml-with-arduino/
categories:
    - 'No classified'
tags:
    - Arduino
    - 'Machine learning'
    - Tensorflow
    - 'Tensorflow Lite'
    - TinyML
---

<img src="{{ site.relative_url }}assets/images/2021/07/ant-947402_1280-1024x682.jpg" alt="Ant" width="900" />

In this article I will show how to build Tensorflow Lite based jelly bears classifier using [Arduino Nano 33 BLE Sense](https://docs.arduino.cc/hardware/nano-33-ble-sense).

Before you will continue reading please watch short introduction: 

https://www.youtube.com/watch?v=dcjEx9qC0o4

Currently a machine learning solution can be deployed not only on very powerful machines containing GPU cards but also on a really small devices. Of course such a devices has a some limitation eg. memory etc. To deploy ML model we need to prepare it. The Tensorflow framework allows you to convert neural networks to Tensorflow Lite which can be installed on the edge devices eg. Arduino Nano. 

[Arduino Nano 33 BLE Sense](https://docs.arduino.cc/hardware/nano-33-ble-sense) is equipped with many sensors that allow for the implementation of many projects eg.:
* Digital microphone
* Digital proximity, ambient light, RGB and gesture sensor
* 3D magnetometer, 3D accelerometer, 3D gyroscope
* Capacitive digital sensor for relative humidity and temperature

Examples which I have used in this project can be found [here](https://create.arduino.cc/projecthub/gilbert-tanner/arduino-nano-33-ble-sense-overview-371c69).

<img src="{{ site.relative_url }}assets/images/2021/07/arduino_all-1024x441.png" alt="Arduino sensors" width="900" />

To simplify device usage I have build [Arduino Lab](https://github.com/qooba/tinyml-arduino) project where you can test and investigate listed sensors directly on the web browser.

The project dependencies are packed into docker image to simplify usage.

Before you start the project you will need to connect Arduino through USB (the Arduino will communicate with docker container through **/dev/ttyACM0**)

```bash
git clone https://github.com/qooba/tinyml-arduino.git
cd tinyml-arduino
./run.server.sh
# in another terminal tab
./run.nginx.sh
# go inside server container 
docker exec -it arduino /bin/bash
./start.sh
```

For each sensor type you can click **Prepare** button which will build and deploy appropriate Arduino code.

---
**NOTE:**
Sometimes you will have to deploy to arduino manually to do this you will need to 
go to arduino container 
```
docker exec -it arduino /bin/bash
cd /arduino
make rgb
```

Here you have complete [Makefile](https://github.com/qooba/tinyml-arduino/blob/main/src/arduino/Makefile) with all types of implemented sensors.

---

You can start observations using **Watch** button.
<img src="{{ site.relative_url }}assets/images/2021/07/TinyMLArduino_pdm-1024x576.jpg" alt="Arduino pdm" width="900" />
<img src="{{ site.relative_url }}assets/images/2021/07/TinyMLArduino_temperature-1024x576.jpg" alt="Arduino temperature" width="900" />
<img src="{{ site.relative_url }}assets/images/2021/07/TinyMLArduino_rgb-1024x576.jpg" alt="Arduino rgb" width="900" />

Now we will build TinyML solution. 
In the first step we will capture training data:
<img src="{{ site.relative_url }}assets/images/2021/07/TinyMLArduino_capture-1024x576.jpg" alt="Arduino capture" width="900" />

The training data will be saved in the csv format. You will need to repeat the proces for each class you will detect.

Captured data will be uploaded to the [Colab Notebook](https://colab.research.google.com/github/arduino/ArduinoTensorFlowLiteTutorials/blob/master/FruitToEmoji/FruitToEmoji.ipynb).
Here I fully base on the project [Fruit identification using Arduino and TensorFlow](https://blog.arduino.cc/2019/11/07/fruit-identification-using-arduino-and-tensorflow/).
In the notebook we train the model using Tensorflow then convert it to Tensorflow Lite and finally encode to hex format (**model.h** header file) which is readable by Arduino.

Now we compile and upload **model.h** header file using drag and drop mechanism.

<img src="{{ site.relative_url }}assets/images/2021/07/TinyMLArduino_upload-1024x576.jpg" alt="Arduino upload" width="900" />

Finally we can classify the jelly bears by the color:

<img src="{{ site.relative_url }}assets/images/2021/07/TinyMLArduino_classify-1024x576.jpg" alt="Arduino classify" width="900" />