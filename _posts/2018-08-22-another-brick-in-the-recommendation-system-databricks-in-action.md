---
id: 19
title: 'Another brick in the &#8230; recommendation system &#8211; Databricks in action'
date: '2018-08-22T17:59:09+02:00'
author: qooba
layout: post
guid: 'http://qooba.net/?p=19'
permalink: /2018/08/22/another-brick-in-the-recommendation-system-databricks-in-action/
categories:
    - 'No classified'
tags:
    - 'collaborative filtering'
    - Datbricks
    - Python
    - recommendations
    - Spark
---

![Brick]({{ site.relative_url }}assets/images/2018/08/wall-450106_640.jpg)

Today I'd like to investigate the [Databricks](https://databricks.com/). I will show how it works and how to prepare simple recommendation system using collaborative filtering algorithm which can be used to help to match the product to the expectations and preferences of the user. Collaborative filtering algorithm is extremely useful when we know the relations (eg ratings) between products and the users but it is difficult to indicate the most significant features. 

## Databricks

First of all, I have to setup the databricks service where I can use [Microsoft Azure Databricks](https://databricks.com/product/azure) or [Databricks on AWS](https://databricks.com/aws) but the best way to start is to use the [Community](https://community.cloud.databricks.com) version. 

### Data

In this example I use the [movielens small dataset](http://files.grouplens.org/datasets/movielens/ml-latest-small.zip) to create recommendations for the movies. After unzipping the package I use the **ratings.csv** file.
On the main page of Databrick click **Upload Data** and put the file.

{% gist 1765d58da9b99801c92e63ecbcec385d uploadfile.png %}

The file will be located on the DBFS (Databrick file system) and will have a path **/FileStore/tables/ratings.csv**. Now I can start model training.

### Notebook

The data is ready thus in the next step I can create the databricks notebook (**new notebook** option on the databricks main page) similar to Jupyter notebook.

Using databricks I can prepare the recommendation in a few steps:

{% gist 1765d58da9b99801c92e63ecbcec385d recommendation.png %}

First of all I read and parse the data, because the data file contains the header additionally I have to cut it. In the next step I split the data into training which will be used to train the model and testing part for model evaluation. 

I can simply create the ratings for each user/product pair but also export user and products (in this case movies) features. The features in general are meaningless factors but deeper analysis and intuition can give them meaning eg movie genre. The number of features is defined by the **rank** parameter in training method (used for model training).
The user/product rating is defined as a scalar product of user and product feature vectors.
This gives us ability to use them outside the databricks eg in relational database prefilter the movies using defined business rules and then order using user/product features.

Finally I have shown how to save the user and product features as a json and put it to Azure blob.