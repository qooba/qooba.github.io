---
id: 38
title: 'Tensorflow meets C# Azure function'
date: '2018-09-10T00:28:37+02:00'
author: qooba
layout: post
guid: 'http://qooba.net/?p=38'
permalink: /2018/09/10/tensorflow-meets-c-azure-function/
categories:
    - 'No classified'
tags:
    - Azure
    - 'Azure functions'
    - 'C#'
    - 'Machine learning'
    - 'Neural networks'
    - Serverless
    - Tensorflow
---

![Meet]({{ site.relative_url }}wp-content/uploads/2018/10/meet_640.png)

Tensorflow meets C# Azure function and … . In this post I would like to show how to deploy tensorflow model with C# Azure function. I will use the [TensorflowSharp](https://github.com/migueldeicaza/TensorFlowSharp) the .NET bindings to the tensorflow library. The [InterceptionInterface](https://github.com/migueldeicaza/TensorFlowSharp/tree/master/Examples/ExampleInceptionInference) will be involved to create http endpoint which will recognize the images. 

## Code

I will start with creating .net core class library and adding TensorFlowSharp package:

```
dotnet new classlib
dotnet add package TensorFlowSharp -v 1.9.0
```

Then create file **TensorflowImageClassification.cs**:

{% gist 19063c20dc664df39511f7d8e6cc1605 TensorflowImageClassification.cs %}

Here I have defined the http entrypoint for the AzureFunction (**Run** method). The **q** query parameter is taken from the url and used as a url of the image which will be recognized. 

The solution will analyze the image using the convolutional neural network arranged with the [Interception architecture](https://arxiv.org/abs/1512.00567).

The function will automatically download the [trained interception model](https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip) thus the function first run will take little bit longer. The model will be saved to the **D:\home\site\wwwroot\\**.

The convolutional neural network graph will be kept in the memory (**graphCache**) thus the function don’t have to read the model every request. On the other hand the input image tensor has to be prepared and preprocessed every time (**ConstructGraphToNormalizeImage**).

Finally I can run command:
```
dotnet publish
```
which will create the package for the function deployment.


## Azure function

To deploy the code I will create the Azure Function (Consumption) with the http trigger. Additionally I will set the function entry point, the **function.json** will be defined as:

{% gist 19063c20dc664df39511f7d8e6cc1605 function.json %}

The kudu will be used to deploy the already prepared package. Additionally I have to deploy the **libtensorflow.dll** from **/runtimes/win7-x64/native** (otherwise the Azure Functions won’t load it). The bin directory should look like:

{% gist 19063c20dc664df39511f7d8e6cc1605 kudu.png %}

Finally I can test the azure function:

{% gist 19063c20dc664df39511f7d8e6cc1605 azure_function.png %}

The function recognize the image and returns the label with the highest probability.
