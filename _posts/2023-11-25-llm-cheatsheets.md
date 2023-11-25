---
id: 743
title: 'Transform Your Coding Journey: Interactive Cheat Sheets with LLM Assistance'
date: '2023-11-25T08:00:00'
author: qooba
layout: post
guid: 'https://blog.qooba.net/?p=743'
permalink: /2023/11/25/cheat-sheets-with-llm/
categories:
    - Cheatsheets
    - LLM
    - AI
    - MachineLearning
    - LanguageModels
    - ArtificialIntelligence
tags:
    - ArtificialIntelligence
    - LLM
    - AI
    - MachineLearning
    - LanguageModels
    - Cheatsheets
---

<img src="{{ site.relative_url }}assets/images/2023/11/lamp.png" alt="all" width="900" />

Cheat sheets are common companions in the journey through programming. 
They are incredibly helpful, offering quick references. 

But what if we could take them a step further? Imagine these cheat sheets not just as static helpers, 
but as dynamic, interactive guides with the power of large language models. 
These enhanced cheat sheets don't just provide information; they interact, they understand, and they assist.
Let's explore how we can make this leap.

Before you will continue reading please watch short introduction: 

<div class="video-container">
    <iframe src="https://www.youtube.com/embed/RDhXRGfeCwk" frameborder="0" allowfullscreen></iframe>
</div>


In the first step I have built Vue web application with responsive cheatsheet layout.

Next, I have brought Python into the browser using the [Pyodide](https://github.com/pyodide/pyodide) library. 
Pyodide is a port of CPython to WebAssembly. 
This means that we can run Python code right in the web browser, 
seamlessly integrating live coding examples and real-time feedback into cheatsheets.

The final, and perhaps the most exciting step, was adding LLM genie, 
our digital assistant. Using the [mlc-llm](https://github.com/mlc-ai/mlc-llm) library, I have embedded a powerful 
large language models into the web application. Currently we can choose and test several models 
like: [RedPajama](https://huggingface.co/togethercomputer), [LLama2](https://ai.meta.com/llama/) or [Mistral](https://docs.mistral.ai/).
First and foremost, the LLM model, 
is designed to run directly in your browser on your device. 
This means that once the LLM is downloaded, all its processing and interactions happen locally,
thus its performance depends on your device capabilities. 
If you want you to test it on my website:  

[https://www.onceuponai.dev/](https://www.onceuponai.dev/)

Here, you can test the interactive cheat sheets and challenge the LLM with your code.



