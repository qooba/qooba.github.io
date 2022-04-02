---
id: 17
title: 'Hello from serverless messanger chatbot'
date: '2018-07-22T23:50:25+02:00'
author: qooba
layout: post
guid: 'http://qooba.net/?p=17'
permalink: /2018/07/22/hello-from-serverless-messanger-chatbot/
categories:
    - Python
tags:
    - AWS
    - 'AWS Lambda'
    - Chatbot
    - Facebook
    - Messanger
    - Python
    - Serverless
    - Wit
---

![Chatbot](http://qooba.net/wp-content/uploads/2018/07/light-bulb-3104355_640.jpg)

**Messanger chatbots** are now becoming more and more popular. They can help us order pizzas, ask about the weather or check the news.

In this article, I would like to show you how to build a simple [messanger chatbot](https://developers.facebook.com/docs/messenger-platform ) in python and pass it on AWS lambda. Additionally use the [wit.ai](https://wit.ai/) service to add to it the natural language understanding functionality and make it more intelligent.

To build the messanger chatbot I will need facebook app and facebook page.

## Facebook page

The whole communication is going through a Facebook page thus I need to create it
[https://www.facebook.com/bookmarks/pages ](https://www.facebook.com/bookmarks/pages)

I will need the **page id** which you can find at the bottom of your page:
https://www.facebook.com/**page_name**/about

## Facebook app

Then I create facebook app.
[https://developers.facebook.com/quickstarts/?platform=web](https://developers.facebook.com/quickstarts/?platform=web)

### Settings

I will copy the **AppId** and **AppSecret** which will be needed in the next steps:


{% gist 13d6ca793723d06f96350625ce0f2daf#file-facebook_appid_appsecret-png %}
https://gist.github.com/qooba/13d6ca793723d06f96350625ce0f2daf#file-facebook_appid_appsecret-png

### Messanger product
Then I will add the messanger product and setup it.
I need to select page we already created and copy generated **access token**.

https://gist.github.com/qooba/13d6ca793723d06f96350625ce0f2daf#file-facebook_app_messanger-png

&nbsp;

### Webhook

Finally I have to setup the webhooks for messanger product
To finish this step I need to setup our chatbot on aws lambda.
I also have to provide the **verify token** which will be used to validate our endpoint.

## AWS Lambda

Now I will prepare my chatbot endpoint. I will setup it on the AWS lambda.

### Trigger

For my chatbot I need to configure API Gateway.
I have to choose security **open** otherwise I won't be able to call it from messanger
https://gist.github.com/qooba/13d6ca793723d06f96350625ce0f2daf#file-aws_lambda_apigateway-png

&nbsp;

### Code

I also need to provide code which will handle the messanger webhook and send response.
I will simply put the code in the online editor.
Let's take a look at the code:

https://gist.github.com/qooba/13d6ca793723d06f96350625ce0f2daf#file-chatbot-py

### Configuration

Bellow I have to setup environment variables:
**verify_token** - verification token (I use [keepass](https://keepass.info/) to generate it) which we will use in webhook setup
**access_token** - value from the messanger webhook page setup
**app_secret** - facebook app secret

https://gist.github.com/qooba/13d6ca793723d06f96350625ce0f2daf#file-aws_lambda_config-png

Now I'm ready to finish the webhook configuration:
https://gist.github.com/qooba/13d6ca793723d06f96350625ce0f2daf#file-facebook_app_events-png

I use api gateway url ass **Callback URL** and **verify_token** I have just generated.

## Natural language undestanding

Messanger give easy way to add natural language undestanding functionality. To add this I simply configure it on messanger product setup page

https://gist.github.com/qooba/13d6ca793723d06f96350625ce0f2daf#file-facebook_app_nlp-png

Here I can choose already trained models but I will go further and I will create custom model.
Messanger will create the new [wit.ai](https://wit.ai) project for me.

On the wit.ai I can simply add some intents (like: **hungry**) and additional information which can be retrieved from the phrase (like: I want some **pizza**)

The messanger/wit integration is very smooth let's analyze the webhook json I get when I put *I want to eat pizza*

https://gist.github.com/qooba/13d6ca793723d06f96350625ce0f2daf#file-webhook-json

After wit integration the **nlp** object was added. Now I can get the recognized intent with some confidence (like: **hungry**) and additional entities (like: **dish**).

Finally I can talk with my chatbot :)

https://gist.github.com/qooba/13d6ca793723d06f96350625ce0f2daf#file-chatbot_hello-png
