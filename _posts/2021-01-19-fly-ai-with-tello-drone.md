---
id: 525
title: 'Fly AI with Tello drone'
date: '2021-01-19T21:00:40+01:00'
author: qooba
layout: post
guid: 'https://qooba.net/?p=525'
permalink: /2021/01/19/fly-ai-with-tello-drone/
categories:
    - 'No classified'
tags:
    - 'Deep learning'
    - Drone
    - 'Machine learning'
    - 'Object Detection'
    - Tensorflow
---

<img src="https://qooba.net/wp-content/uploads/2021/01/hot-air-balloons-1867279_1280-1024x678.jpg" alt="balloons" width="900" />

The popularity of drones and the area of their application is becoming greater each year. 

In this article I will show how to programmatically control [Tello Ryze drone](https://www.ryzerobotics.com/), capture camera video and detect objects using [Tensorflow](https://www.tensorflow.org/). I have packed the whole solution into [docker images](https://hub.docker.com/repository/docker/qooba/aidrone) (the backend and Web App UI are in separate images) thus you can simply run it.


The project code is available on my github [https://github.com/qooba/aidrone](https://github.com/qooba/aidrone)
You can also use ready docker image: [https://hub.docker.com/repository/docker/qooba/aidrone](https://hub.docker.com/repository/docker/qooba/aidrone)

Before you will continue reading please watch short introduction: 
https://youtu.be/g8oZ8ltRArY


# Architecture

<img src="https://qooba.net/wp-content/uploads/2021/01/AIDrone-1024x590.png" alt="architecture diagram" width="900" />

The application will use two network interfaces. 
The first will be used by the python backend to connect the the Tello wifi to send the commands and capture video stream. In the backend layer I have used the [DJITelloPy](https://github.com/damiafuentes/DJITelloPy/) library which covers all required tello move commands and video stream capture.
To efficiently show the video stream in the browser I have used the WebRTC protocol and [aiortc](https://github.com/aiortc/aiortc) library. Finally I have used the [Tensorflow 2.0 object detection](https://github.com/tensorflow/models/tree/master/research/object_detection) with pretrained [SSD ResNet50 model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). 

The second network interface will be used to expose the [Web Vue application](https://vuejs.org/). 
I have used [nginx](https://www.nginx.com/) to serve the frontend application

# Application


<img src="https://qooba.net/wp-content/uploads/2021/01/AIDronGifsMove.gif" alt="drone controls" width="900" />

Using Web interface you can control the Tello movement where you can:
* start video stream
* stop video stream
* takeoff - which starts Tello flight
* land 
* up 
* down 
* rotate left
* rotate right
* forward 
* backward
* left 
* right


In addition using draw detection switch you can turn on/off the detection boxes on the captured video stream (however this introduces a delay in the video thus it is turned off by default). Additionally I send the list of detected classes through web sockets which are also displayed.

<img src="https://qooba.net/wp-content/uploads/2021/01/AIDronGifsDetection.gif" alt="drone detection" width="900" />

As mentioned before I have used the pretrained model thus It is good idea to train your own model to get better results for narrower and more specific class of objects. 

Finally the whole solution is packed into docker images thus you can simply start it using commands:
```bash
docker network create -d bridge app_default
docker run --name tello --network app_default --gpus all -d --rm -p 8890:8890 -p 8080:8080 -p 8888:8888 -p 11111:11111/udp  qooba/aidrone /bin/bash -c "python3 drone.py"
docker run -d --rm --network app_default --name nginx -p 80:80 -p 443:443 qooba/aidrone:front
```

***To use GPU additional nvidia drivers (included in the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)) are needed.***