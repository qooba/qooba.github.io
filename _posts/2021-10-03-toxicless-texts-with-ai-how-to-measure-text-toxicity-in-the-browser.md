---
id: 708
title: 'Toxicless texts with AI &#8211; how to measure text toxicity in the browser'
date: '2021-10-03T19:04:44+02:00'
author: qooba
layout: post
guid: 'https://qooba.net/?p=708'
permalink: /2021/10/03/toxicless-texts-with-ai-how-to-measure-text-toxicity-in-the-browser/
categories:
    - 'No classified'
tags:
    - 'Natural Language Processing'
    - 'Natural Language Understanding'
    - NLP
    - NLU
    - Tensorflow
    - 'Tensorflow Lite'
    - WebAssembly
---

<img src="https://qooba.net/wp-content/uploads/2021/10/internet.jpg" alt="internet" width="900" />

In this article I will show how to measure comments toxicity using Machine Learning models.

Before you will continue reading please watch short introduction: 

https://www.youtube.com/watch?v=AECV2qa0Kaw

Hate, rude and toxic comments are common problem in the internet which affects many people. Today, we will prepare neural network, which detects comments toxicity, directly in the browser. The goal is to create solution which will detect toxicity in the realtime and warn the user during writing, which can discourage from writing toxic comments.

To do this, we will train the [tensorflow lite model](https://www.tensorflow.org/lite), which will run in the browser using WebAssembly backend. The [WebAssembly (WASM)](https://webassembly.org/) allows running C, C++ or RUST code at native speed. Thanks to this, prediction performance will be better than running it using javascript tensorflowjs version.
Moreover, we can serve the model, on the static page, with no additional backend servers required.

<img src="https://qooba.net/wp-content/uploads/2021/10/AIToxicity.00.png" alt="web assembly" width="900" />

To train the model, we will use the [Kaggle Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) training data,
which contains the labeled comments, with toxicity types:
* toxic
* severe_toxic
* obscene
* threat
* insult
* identity_hate

<img src="https://qooba.net/wp-content/uploads/2021/10/AIToxicity.01.png" alt="data set" width="900" />

Our model, will only classify, if the text is toxic, or not. Thus we need to start with preprocessing training data. Then we will use the [tensorflow lite model maker library](https://www.tensorflow.org/lite/tutorials/model_maker_text_classification).
We will also use the *Averaging Word Embedding* specification which will create words embeddings and dictionary mappings using training data thus we can train the model in the different languages. 
The *Averaging Word Embedding* specification based model will be small ```<1MB```. 
If we have small dataset we can use the pretrained embeddings. We can choose [*MobileBERT*](https://arxiv.org/pdf/2004.02984.pdf) or [*BERT-Base*](https://arxiv.org/pdf/1810.04805.pdf) specification.
In this case models will much more bigger ```25MB w/ quantization 100MB w/o quantization``` for 
*MobileBERT* and ```300MB``` for *BERT-Base* (based on: https://www.tensorflow.org/lite/tutorials/model_maker_text_classification#choose_a_model_architecture_for_text_classifier)

<img src="https://qooba.net/wp-content/uploads/2021/10/AIToxicity4_2.jpg" alt="train" width="900" />

Using simple model architecture (*Averaging Word Embedding*), we can achieve about nighty five percent accuracy, and small model size, appropriate 
for the web browser, and web assembly. 

<img src="https://qooba.net/wp-content/uploads/2021/10/AIToxicity5_1.jpg" alt="tensorflow lite" width="900" />

Now, let's prepare the non-toxic forum web application, where we can write the comments.
When we write non-toxic comments, the model won't block it.
On the other hand, the toxic comments will be blocked, and the user warned.

Of course, this is only client side validation, which can discourage users, from writing toxic comments.

<img src="https://qooba.net/wp-content/uploads/2021/10/TextToxicity.gif" alt="web application" width="900" />

To run the example simply clone git repository and run simple server to serve the static page:
``` bash 
git clone https://github.com/qooba/ai-toxicless-texts.git
cd ai-toxicless-texts
python3 -m http.server
```

The code to for preparing data, training and exporting model is here:
[https://github.com/qooba/ai-toxicless-texts/blob/master/Model_Maker_Toxicity.ipynb](https://github.com/qooba/ai-toxicless-texts/blob/master/Model_Maker_Toxicity.ipynb)