---
id: 271
title: 'FastAI with TensorRT on Jetson Nano'
date: '2020-05-10T15:54:46+02:00'
author: qooba
layout: post
guid: 'http://qooba.net/?p=271'
permalink: /2020/05/10/fastai-with-tensorrt-on-jetson-nano/
categories:
    - 'No classified'
tags:
    - FastAI
    - JetsonNano
    - 'Machine learning'
    - Python
    - PyTorch
    - TensorRT
---

![DIV]({{ site.relative_url }}assets/images/2020/05/cheetah-2859581_640.jpg)

IoT and AI are the hottest topics nowadays which can meet on [Jetson Nano device](https://www.nvidia.com/pl-pl/autonomous-machines/embedded-systems/jetson-nano/). 
In this article I'd like to show how to use [FastAI](https://www.fast.ai/) library, which is built on the top of the [PyTorch](https://pytorch.org/) on Jetson Nano. Additionally I will show how to optimize the [FastAI](https://www.fast.ai/) model for the usage with [TensorRT](https://developer.nvidia.com/tensorrt).

You can find the code on [https://github.com/qooba/fastai-tensorrt-jetson.git](https://github.com/qooba/fastai-tensorrt-jetson.git).

# 1. Training

Although the Jetson Nano is equipped with the GPU it should be used as a inference device rather than for training purposes. Thus I will use another PC with the GTX 1050 Ti for the training.

Docker gives flexibility when you want to try different libraries thus I will use the image which contains the complete environment.

Training environment Dockerfile:
``` Dockerfile
FROM nvcr.io/nvidia/tensorrt:20.01-py3
WORKDIR /
RUN apt-get update && apt-get -yq install python3-pil
RUN pip3 install jupyterlab torch torchvision
RUN pip3 install fastai
RUN DEBIAN_FRONTEND=noninteractive && apt update && apt install curl git cmake ack g++ tmux -yq
RUN pip3 install ipywidgets && jupyter nbextension enable --py widgetsnbextension
CMD ["sh","-c", "jupyter lab --notebook-dir=/opt/notebooks --ip='0.0.0.0' --port=8888 --no-browser --allow-root --NotebookApp.password='' --NotebookApp.token=''"]
```

*To use GPU additional nvidia drivers (included in the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)) are needed.*

If you don't want to build your image simply run:
``` bash
docker run --gpus all  --name jupyter -d --rm -p 8888:8888 -v $(pwd)/docker/gpu/notebooks:/opt/notebooks qooba/fastai:1.0.60-gpu
```

Now you can use [pets.ipynb](https://github.com/qooba/fastai-tensorrt-jetson/blob/master/docker/gpu/notebooks/pets.ipynb) notebook (the code is taken from [lesson 1 FastAI course](https://course.fast.ai/)) to train and export pets classification model.

``` python
from fastai.vision import *
from fastai.metrics import error_rate

# download dataset
path = untar_data(URLs.PETS)
path_anno = path/'annotations'
path_img = path/'images'
fnames = get_image_files(path_img)

# prepare data 
np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$'
bs = 16
data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs).normalize(imagenet_stats)

# prepare model learner
learn = cnn_learner(data, models.resnet34, metrics=error_rate)

# train 
learn.fit_one_cycle(4)

# export
learn.export('/opt/notebooks/export.pkl')
```

Finally you get pickled pets model (**export.pkl**). 

# 2. Inference (Jetson Nano)

The Jetson Nano device with [Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit) already comes with the docker thus I will use it to setup the inference environment.

I have used the base image [nvcr.io/nvidia/l4t-base:r32.2.1](https://ngc.nvidia.com/catalog/containers/nvidia:l4t-base) and installed the [pytorch and torchvision](https://forums.developer.nvidia.com/t/pytorch-for-jetson-nano-version-1-5-0-now-available/72048).
If you have [JetPack 4.4 Developer Preview](https://developer.nvidia.com/embedded/jetpack) you can skip this steps and start with the base image [nvcr.io/nvidia/l4t-pytorch:r32.4.2-pth1.5-py3](https://ngc.nvidia.com/catalog/containers/nvidia:l4t-pytorch).

The FastAI installation on Jetson is more problematic because of the **blis** package. Finally I have found the solution [here](https://github.com/explosion/cython-blis/issues/9). 

Additionally I have installed [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt.git) package which converts PyTorch model to TensorRT. 

Finally I have used the tensorrt from the JetPack which can be found in
*/usr/lib/python3.6/dist-packages/tensorrt* .

The final Dockerfile is:
``` Dockerfile
FROM nvcr.io/nvidia/l4t-base:r32.2.1
WORKDIR /
# install pytorch 
RUN apt update && apt install -y --fix-missing make g++ python3-pip libopenblas-base
RUN wget https://nvidia.box.com/shared/static/ncgzus5o23uck9i5oth2n8n06k340l6k.whl -O torch-1.4.0-cp36-cp36m-linux_aarch64.whl
RUN pip3 install Cython
RUN pip3 install numpy torch-1.4.0-cp36-cp36m-linux_aarch64.whl
# install torchvision
RUN apt update && apt install libjpeg-dev zlib1g-dev git libopenmpi-dev openmpi-bin -yq
RUN git clone --branch v0.5.0 https://github.com/pytorch/vision torchvision
RUN cd torchvision && python3 setup.py install
# install fastai
RUN pip3 install jupyterlab
ENV TZ=Europe/Warsaw
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && apt update && apt -yq install npm nodejs python3-pil python3-opencv
RUN apt update && apt -yq install python3-matplotlib
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git /torch2trt && mv /torch2trt/torch2trt /usr/local/lib/python3.6/dist-packages && rm -r /torch2trt
COPY tensorrt /usr/lib/python3.6/dist-packages/tensorrt
RUN pip3 install --no-deps fastai
RUN git clone https://github.com/fastai/fastai /fastai
RUN apt update && apt install libblas3 liblapack3 liblapack-dev libblas-dev gfortran -yq
RUN curl -LO https://github.com/explosion/cython-blis/files/3566013/blis-0.4.0-cp36-cp36m-linux_aarch64.whl.zip && unzip blis-0.4.0-cp36-cp36m-linux_aarch64.whl.zip && rm blis-0.4.0-cp36-cp36m-linux_aarch64.whl.zip
COPY blis-0.4.0-cp36-cp36m-linux_aarch64.whl .
RUN pip3 install scipy pandas blis-0.4.0-cp36-cp36m-linux_aarch64.whl spacy fastai scikit-learn
CMD ["sh","-c", "jupyter lab --notebook-dir=/opt/notebooks --ip='0.0.0.0' --port=8888 --no-browser --allow-root --NotebookApp.password='' --NotebookApp.token=''"]
```

As before you can skip the docker image build and use ready image:
```
docker run --runtime nvidia --network app_default --name jupyter -d --rm -p 8888:8888 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix -v $(pwd)/docker/jetson/notebooks:/opt/notebooks qooba/fastai:1.0.60-jetson
```

Now we can open jupyter notebook on jetson and move pickled model file **export.pkl** from PC. 
The notebook [jetson_pets.ipynb](https://github.com/qooba/fastai-tensorrt-jetson/blob/master/docker/jetson/notebooks/jetson_pets.ipynb) show how to load the model. 

``` python
import torch
from torch2trt import torch2trt
from fastai.vision import *
from fastai.metrics import error_rate

learn = load_learner('/opt/notebooks/')
learn.model.eval()
model=learn.model

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')
```

Additionally we can optimize the model using [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt) package:

``` python
x = torch.ones((1, 3, 224, 224)).cuda()
model_trt = torch2trt(learn.model, [x])
```

Let's prepare example input data:
``` python
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)
```

Finally we can run prediction for PyTorch and TensorRT model:
``` python
x=input_batch
y = model(x)
y_trt = model_trt(x)
```

and compare PyTorch and TensorRT performance:
``` python
def prediction_time(model, x):
    import time
    times = []
    for i in range(20):
        start_time = time.time()
        y_trt = model(x)
    
        delta = (time.time() - start_time)
        times.append(delta)
    mean_delta = np.array(times).mean()
    fps = 1/mean_delta
    print('average(sec):{},fps:{}'.format(mean_delta,fps))

prediction_time(model,x)
prediction_time(model_trt,x)
```

where for:
* PyTorch - average(sec):0.0446, fps:22.401
* TensorRT - average(sec):0.0094, fps:106.780

The TensorRT model is almost 5 times faster thus it is worth to use [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt). 

# References

[1] Top image  <a href="https://pixabay.com/pl/users/DrZoltan-6737770/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=2859581"> DrZoltan</a> from <a href="https://pixabay.com/pl/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=2859581"> Pixabay</a>
 