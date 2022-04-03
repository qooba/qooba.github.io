---
id: 613
title: 'Flink with AI &#8211; how to use Flink with MLflow model in Jupyter Notebook'
date: '2021-04-12T01:10:46+02:00'
author: qooba
layout: post
guid: 'https://qooba.net/?p=613'
permalink: /2021/04/12/flink-with-ai-how-to-use-flink-with-mlflow-model-in-jupyter-notebook/
categories:
    - 'No classified'
tags:
    - Flink
    - Kafka
    - 'Machine learning'
    - MLflowflow
---

<img src="{{ site.relative_url }}assets/images/2021/04/animal-4501226_1920-1024x683.jpg" alt="squirrel" width="900" />

In this article I will show how to process streams with Apache Flink and MLflow model

Before you will continue reading please watch short introduction: 
https://www.youtube.com/watch?v=DJPv1zGXtkU


[Apache Flink](https://flink.apache.org/) allows for an efficient and scalable way of processing streams. It is a distributed processing engine which supports multiple sources like: [Kafka](https://kafka.apache.org/), [NiFi](https://nifi.apache.org/) and many others
(if we need custom, we can create them ourselves).

Apache Flink also provides the framework for defining streams operations in languages like:
Java, Scala, Python and SQL.

To simplify the such definitions we can use Jupyter Notebook as a interface. Of course we can write in Python using PyFlink library but we can make it even easier using writing [jupyter notebook extension ("magic words")](https://ipython.readthedocs.io/en/stable/config/custommagics.html).

Using Flink extension ([magic.ipynb](https://github.com/qooba/flink-with-ai/blob/main/notebooks/magic/magic.ipynb)) we can simply use Flink SQL sql syntax directly in Jupyter Notebook.

To use the extesnions we need to load it:
```
%reload_ext flinkmagic
```

Then we need to initialize the Flink StreamEnvironment:
```
%flink_init_stream_env
```

Now we can use the SQL code for example:

### FileSystem connector:
```
%%flink_execute_sql
CREATE TABLE MySinkTable (
    word varchar,
    cnt bigint) WITH (
        'connector.type' = 'filesystem',
        'format.type' = 'csv',
        'connector.path' = '/opt/flink/notebooks/data/word_count_output1')
```

### MySQL connector:
```
%%flink_execute_sql
CREATE TABLE MySinkDbSmsTable (
    smstext varchar,
    smstype varchar) WITH (
        'connector.type' = 'jdbc',
        'connector.url' = 'jdbc:mysql://mysql:3306/test',
        'connector.table' = 'sms',
        'connector.driver' = 'com.mysql.jdbc.Driver',
        'connector.write.flush.interval' = '10',
        'connector.username' = 'root',
        'connector.password' = 'my-secret-pw')
```

### Kafka connector:
```
%%flink_execute_sql
CREATE TABLE MySourceKafkaTable (word varchar) WITH (
    'connector.type' = 'kafka',
    'connector.version' = 'universal',
    'connector.topic' = 'test',
    'connector.startup-mode' = 'latest-offset',
    'connector.properties.bootstrap.servers' = 'kafka:9092',
    'connector.properties.group.id' = 'test',
    'format.type' = 'csv'
        )
```

The magic keyword will automatically execute SQL in existing StreamingEnvironment.

Now we can apply the Machine Learning model. In plain Flink we can use UDF function defined
in python but we will use MLflow model which wraps the ML frameworks (like PyTorch, Tensorflow, Scikit-learn etc.). Because MLflow expose homogeneous interface we can 
create another "jupyter magic" which will automatically load MLflow model as a Flink function.

```
%flink_mlflow "SPAM_CLASSIFIER" "/mlflow/mlruns/2/64a89b0a6b7346498316bfae4c298535/artifacts/model" "[DataTypes.STRING()]" "DataTypes.STRING()"
```

Now we can simply write Flink SQL query:
```
%%flink_sql_query
SELECT word as smstext, SPAM_CLASSIFIER(word) as smstype FROM MySourceKafkaTable
```

which in our case will fetch kafka events and classify it using MLflow spam classifier. The 
results will be displayed in the realtime in the Jupyter Notebook as a events DataFrame.

If we want we can simply use other python libraries (like matplotlib and others) to create 
graphical representation of the results eg. [pie chart](https://github.com/qooba/flink-with-ai/blob/main/notebooks/flink_magic.ipynb).

You can find the whole code including: Flink examples, extension and [Dockerfiles](https://github.com/qooba/flink-with-ai/blob/main/docker/flink/Dockerfile) here:
[https://github.com/qooba/flink-with-ai](https://github.com/qooba/flink-with-ai).

You can also use docker image: **qooba/flink:dev** to test and run notebooks inside.
Please check the [run.sh](https://github.com/qooba/flink-with-ai/blob/main/run.sh)
where you have all components (Kafka, MySQL, Jupyter with Flink, MLflow repository).