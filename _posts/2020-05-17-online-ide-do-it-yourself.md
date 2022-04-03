---
id: 314
title: 'Online IDE ? &#8211; do it yourself'
date: '2020-05-17T00:28:56+02:00'
author: qooba
layout: post
guid: 'http://qooba.net/?p=314'
permalink: /2020/05/17/online-ide-do-it-yourself/
categories:
    - 'No classified'
tags:
    - 'Jupyter Notebook'
    - Python
    - vim
---

![DIV]({{ site.relative_url }}assets/images/2020/05/computer-1836330_640.png)

[Jupyter Notebook](https://jupyter.org/) is one of the most useful tool for data exploration, machine learning and fast prototyping. There are many plugins and projects which make it even more powerful:
* [jupyterlab-git](https://github.com/jupyterlab/jupyterlab-git)
* [nbdev](https://github.com/fastai/nbdev)
* [jupyter debugger](https://blog.jupyter.org/a-visual-debugger-for-jupyter-914e61716559)

But sometimes you simply need IDE ...

One of my favorite text editor is vim. It is lightweight, fast and with [appropriate plugins](https://github.com/jarolrod/vim-python-ide) it can be used as a IDE. 
Using Dockerfile you can build jupyter environment with fully equipped vim:
``` bash
FROM continuumio/miniconda3
RUN apt update && apt install curl git cmake ack g++ python3-dev vim-youcompleteme tmux -yq
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/qooba/vim-python-ide/master/setup.sh)"
RUN conda install xeus-python jupyterlab jupyterlab-git -c conda-forge
RUN jupyter labextension install @jupyterlab/debugger @jupyterlab/git
RUN pip install nbdev
RUN echo "alias ls='ls --color=auto'" >> /root/.bashrc
CMD bin/bash
```

Now you can run the image:
``` bash
docker run --name jupyter -d --rm -p 8888:8888 -v $(pwd)/jupyter:/root/.jupyter -v $(pwd)/notebooks:/opt/notebooks qooba/miniconda3 /bin/bash -c "jupyter lab --notebook-dir=/opt/notebooks --ip='0.0.0.0' --port=8888 --no-browser --allow-root --NotebookApp.password='' --NotebookApp.token=''"
```

In the **jupyter lab** start **terminal session**, run **bash** (it works better in bash) and then **vim**.
The online IDE is ready:

![DIV]({{ site.relative_url }}assets/images/2020/05/jupyter_ide.gif)

# References

[1] Top image  <a href="https://pixabay.com/pl/users/Boskampi-3788146/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=1836330"> Boskampi</a> from <a href="https://pixabay.com/pl/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=1836330"> Pixabay</a>