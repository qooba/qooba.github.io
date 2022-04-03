---
id: 478
title: 'AI Scissors &#8211; sharp cut with neural networks'
date: '2020-09-11T01:43:06+02:00'
author: qooba
layout: post
guid: 'https://qooba.net/?p=478'
permalink: /2020/09/11/ai-scissors-sharp-cut-with-neural-networks/
categories:
    - 'No classified'
tags:
    - 'Deep learning'
    - 'Machine learning'
    - 'Neural networks'
    - PyTorch
---

<img src="{{ site.relative_url }}assets/images/2020/09/scissors-1024x682.jpg" alt="scissors" width="900" />


Cutting photos background is one of the most tedious graphical task. In this article will show how to simplify it using neural networks. 

I will use U[latex]^2[/latex]-Net networks which are described in detail in the [arxiv article](https://arxiv.org/pdf/2005.09007.pdf) and python library [rembg](https://github.com/danielgatis/rembg) to create ready to use drag and drop web application which you can use running docker image.  

The project code is available on my github [https://github.com/qooba/aiscissors](https://github.com/qooba/aiscissors)
You can also use ready docker image: [https://hub.docker.com/repository/docker/qooba/aiscissors](https://hub.docker.com/repository/docker/qooba/aiscissors)

Before you will continue reading please watch quick introduction:

<div class="video-container">
    <iframe src="https://www.youtube.com/embed/LM0tQReBgNs" frameborder="0" allowfullscreen></iframe>
</div>

# Neural network

To correctly remove the image background we need to select the most visually attractive objects in an image which is covered by Salient Object Detection (SOD). To connect a low memory and computation cost with competitive results against state of art methods the novel U[latex]^2[/latex]-Net architecture will be used. 

[U-Net convolutional networks](https://arxiv.org/pdf/1505.04597.pdf) have characteristic U shape with symmetric encoder-decoder structure. At each encoding stage the feature maps are downsampled ([torch.nn.MaxPool2d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)) and then upsampled at each decoding
stage ([torch.nn.functional.upsample](https://pytorch.org/docs/stable/nn.functional.html#upsample)). Downsample features are transferred and concatenated with upsample features using residual connections. 

U[latex]^2[/latex]-Net network uses two-level nested U-structure where the main architecture is a U-Net like encoder-decoder and each stage contains residual U-block. Each residual U-block repeats donwsampling/upsampling procedures which are also connected using residual connections.  

<img src="{{ site.relative_url }}assets/images/2020/09/u2net.gif" alt="neural network architecture" width="900" />


Nested U-structure extracts and aggregates the features at each level and enables to capture local and global information from shallow and deep layers.

The U[latex]^2[/latex]-Net architecture is precisely described in [arxiv article](https://arxiv.org/pdf/2005.09007.pdf). Moreover we can go through the pytorch model definition of [U2NET](https://github.com/NathanUA/U-2-Net/blob/0b27f5cc958bac88825b1001f8245f663faeb1b8/model/u2net.py#L319) and [U2NETP](https://github.com/NathanUA/U-2-Net/blob/0b27f5cc958bac88825b1001f8245f663faeb1b8/model/u2net.py#L424).

Additionally the authors also shared the pretrained models: [U2NET (176.3MB)](https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view?usp=sharing) and [U2NETP (4.7 MB)](https://drive.google.com/file/d/1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy/view?usp=sharing).

The lighter U2NETP version is only 4.7 MB thus it can be used in mobile applications. 

# Web application

The neural network is wrapped with [rembg library](https://github.com/danielgatis/rembg) which automatically download pretrained networks and gives simple python api. To simplify the usage I have decided to create drag and drop web application ([https://github.com/qooba/aiscissors](https://github.com/qooba/aiscissors)) 

In the application you can drag and the drop the image and then compare image with and without background side by side. 

<img src="{{ site.relative_url }}assets/images/2020/09/app.gif" alt="web application" width="900" />

You can simply run the application using docker image:
```bash
docker run --name aiscissors -d -p 8000:8000 --rm -v $(pwd)/u2net_models:/root/.u2net qooba/aiscissors 
```

if you have GPU card you can use it:
```bash
docker run --gpus all  --name aiscissors -d -p 8000:8000 --rm -v $(pwd)/u2net_models:/root/.u2net qooba/aiscissors 
```

***To use GPU additional nvidia drivers (included in the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)) are needed.***

When you run the container the pretrained models are downloaded thus I have mount local directory **u2net_models** to **/root/.u2net** to avoid download each time I run the container.

# References

[https://arxiv.org/pdf/2005.09007.pdf](https://arxiv.org/pdf/2005.09007.pdf)

[https://github.com/NathanUA/U-2-Net](https://github.com/NathanUA/U-2-Net)

[https://github.com/danielgatis/rembg](https://github.com/danielgatis/rembg)

*U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection*, Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Dehghan, Masood and Zaiane, Osmar and Jagersand, Martin **Pattern Recognition 106 107404 (2020)**


