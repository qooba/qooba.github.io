---
id: 745
title: 'Text transmutation - recipe for semantic search with embeddings'
date: '2023-12-31T08:00:00'
author: qooba
layout: post
guid: 'https://blog.qooba.net/?p=745'
permalink: /2023/12/31/text-transmutation-recipe-for-semantic-search-with-embeddings/
categories:
    - textembeddings
    - LLM
    - AI
    - MachineLearning
    - LanguageModels
    - ArtificialIntelligence
    - WebAssembly
    - Rust
tags:
    - ArtificialIntelligence
    - LLM
    - AI
    - MachineLearning
    - LanguageModels
    - TextEmbeddings
---

<img src="{{ site.relative_url }}assets/images/2023/12/alchemist.png" alt="all" width="900" />

In the rapidly evolving area of data science and natural language processing (NLP), 
the ability to intelligently understand and process textual information is crucial.
In this article I will show how to create a semantic search aplication 
using the [Candle ML](https://github.com/huggingface/candle) framework written in Rust, 
coupled with the [E5 model](https://huggingface.co/intfloat/e5-small-v2) for embedding generation.

Before you will continue reading please watch short introduction: 

<div class="video-container">
    <iframe src="https://www.youtube.com/embed/-eHItEq6A6o" frameborder="0" allowfullscreen></iframe>
</div>

Text embeddings are at the heart of modern natural language processing (NLP). 
They are the result of transforming textual data into a numerical 
form that machines can understand.

<img src="{{ site.relative_url }}assets/images/2023/12/TextTransmutation04.jpg" alt="embeddings" width="900" />

To calculate embeddings I will use the [E5 model](https://huggingface.co/intfloat/e5-small-v2) ([arxiv2212.03533](https://arxiv.org/pdf/2212.03533.pdf))
from Hugging Face to generate text embeddings. 

E5 name comes from embeddings from bidirectional encoder representations.
Model was trained on Colossal Clean text Pairs from heterogeneous semi-structured 
data sources like: Reddit (post, comment), Stackexchange (question, upvoted answer),
English Wikipedia (entity name + section title, passage), Scientific papers (title, abstract), Common Crawl (title, passage), and others. 

To run the E5 model I will use the [Candle ML](https://github.com/huggingface/candle) framework written in Rust.
Candle supports a wide range of ML models including: Whisper, LLama2, Mistral, Stable Diffusion and others.
Moreover we can simply compile and use Candle library inside WebAssembly to calculate 
text embeddings.

To demonstrate the power of these embeddings, I have created a simple search application.
The application contains two parts: rust code which is compiled to WebAssembly and 
Vue web application. 


<img src="{{ site.relative_url }}assets/images/2023/12/TextTransmutation08.jpg" alt="webapp" width="900" />

The rust code is based on the candle Web Assembly example and expose model struct which 
loads the E5 model and calculates embeddings.
Compiled rust struct is used in the Vue typescript webworker. 

The web application reads example recipes and calculates embeddings for each. 

When user inputs a text application calculates embedding and search the recipe 
from the list that matches the best, the cosine similarity is used for this purpose.

Cosine similarity measures the cosine of the angle between two vectors, 
offering a way to judge how similar two texts are in their semantic content.

<img src="{{ site.relative_url }}assets/images/2023/12/TextTransmutation_cosine.png" alt="cosine similarity" width="900" />

For handling larger datasets, it becomes impractical to compute cosine similarity for each phrase individually due to scalability issues. 
In such cases, utilizing a vector database is a more efficient approach.

Application code is available here: [https://github.com/onceuponai-dev/stories-text-transmutation](https://github.com/onceuponai-dev/stories-text-transmutation)
The rust part is based on [Candle example](https://github.com/huggingface/candle/tree/main/candle-wasm-examples/bert)

You can also quickly test model on: [https://stories.onceuponai.dev/stories-text-transmutation/](https://stories.onceuponai.dev/stories-text-transmutation/)



