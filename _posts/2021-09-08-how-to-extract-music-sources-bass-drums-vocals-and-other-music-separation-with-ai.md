---
id: 698
title: 'How to extract music sources: bass, drums, vocals and other ? &#8211; music separation with AI'
date: '2021-09-08T01:44:08+02:00'
author: qooba
layout: post
guid: 'https://qooba.net/?p=698'
permalink: /2021/09/08/how-to-extract-music-sources-bass-drums-vocals-and-other-music-separation-with-ai/
categories:
    - 'No classified'
tags:
    - 'Machine learning'
    - Music
    - 'Music separation'
    - 'Neural networks'
    - PyTorch
---

<img src="{{ site.relative_url }}assets/images/2021/09/sound-4872773_1280.jpg" alt="calculator" width="900" />

In this article I will show how we can extract music sources: bass, drums, vocals and other accompaniments using neural networks.

Before you will continue reading please watch short introduction: 

<div class="video-container">
    <iframe src="https://www.youtube.com/embed/uzerSt4UkYc" frameborder="0" allowfullscreen></iframe>
</div>

Separation of individual instruments from arranged music is another area where machine learning 
algorithms could help. [Demucs](https://github.com/facebookresearch/demucs) solves this problem using neural networks.

The trained model ([https://arxiv.org/pdf/1909.01174v1.pdf](https://arxiv.org/pdf/1909.01174v1.pdf)) use U-NET architecture which contains two parts encoder and decoder. 
On the encoder input we put the original track and after processing we get bass, drums, vocals and other accompaniments at the decoder output. 

The encoder, is connected to the decoder, 
through additional LSTM layer,
as well as residual connections between subsequent layers.

<img src="{{ site.relative_url }}assets/images/2021/09/UNET_colors.png" alt="neural network architecture" width="900" />

Ok, we have neural network architecture but what about the training data ? 
This is another difficulty which can be handled by the unlabeled data remixing pipeline.

We start with another classifier, which can find the parts of music, 
which do not contain the specific instruments, for example drums.
Then, we mix it with well known drums signal, and separate the tracks
using the model. 

Now we can compare, the separation results, with known drums track and mixture of other instruments. 

According to this, we can calculate the loss (L1 loss), and use it during the training. 

Additionally, we set different loss weights, for known track and the other. 

<img src="{{ site.relative_url }}assets/images/2021/09/AIMusicSeparation.00.jpeg" alt="training data" width="900" />

The whole UI is kept in the docker image thus you can simply try it:
```bash
#for CPU
docker run --name aiaudioseparation -it -p 8000:8000 -v $(pwd)/checkpoints:/root/.cache/torch/hub/checkpoints --rm qooba/aimusicseparation

#for GPU
docker run --name aiaudioseparation --gpus all -it -p 8000:8000 -v $(pwd)/checkpoints:/root/.cache/torch/hub/checkpoints --rm qooba/aimusicseparation
```

<img src="{{ site.relative_url }}assets/images/2021/09/AIMusicSeparation.01.jpeg" alt="web UI" width="900" />



