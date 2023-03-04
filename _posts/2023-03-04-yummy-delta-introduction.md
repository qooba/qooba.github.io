---
id: 739
title: 'Discover a Delicious Way to Use Delta Lake! Yummy Delta - #1 Introduction'
date: '2023-03-04T11:00:00'
author: qooba
layout: post
guid: 'https://blog.qooba.net/?p=739'
permalink: /2023/03/04/speedup-mlflow-models-serving-with-rust/
categories:
    - DeltaLake
    - MLOps
    - rust
tags:
    - DeltaLake
    - MLOps
    - rust
---

<img src="{{ site.relative_url }}assets/images/2023/03/YummyDelta_intro.png" alt="yummy delta" width="900" />

[Delta lake](https://delta.io/) is an open source storage framework for building lake house architectures
on top of data lakes.

Additionally it brings reliability to data lakes with features like:
ACID transactions, scalable metadata handling, schema enforcement, time travel and many more.

Before you will continue reading please watch short introduction: 

<div class="video-container">
    <iframe src="https://www.youtube.com/embed/7KoeQ7vA6Q0" frameborder="0" allowfullscreen></iframe>
</div>

Delta lake can be used with compute engines like Spark, Flink, Presto, Trino and Hive. It also
has API for Scala, Java, Rust , Ruby and Python.

<img src="{{ site.relative_url }}assets/images/2023/03/YummyDelta.00.jpeg" alt="delta lake" width="900" />

To simplify integrations with delta lake I have built a REST API layer called Yummy Delta.

Which abstracts multiple delta lake tables providing operations like: creating new delta table, 
writing and querying, but also optimizing and vacuuming. 
I have coded an overall solution in rust based on the [delta-rs](https://github.com/delta-io/delta-rs) project.

Delta lake keeps the data in parquet files which is an open source, 
column-oriented data file format.

Additionally it writes the metadata in the transaction log, 
json files containing information about all performed operations.

The transaction log is stored in the delta lake `_delta_log` subdirectory.

<img src="{{ site.relative_url }}assets/images/2023/03/YummyDelta.01.jpeg" alt="delta lake" width="900" />

For example, every data write will create a new parquet file. 
After data write is done a new transaction log file will be created which finishes the transaction.
Update and delete operations will be conducted in a similar way.
On the other hand when we read data from delta lake at the first stage transaction 
files are read and then according to the transaction data appropriate parquet files are loaded.

Thanks to this mechanism the delta lake guarantees ACID transactions.

There are several delta lake integrations and one of them is [delta-rs](https://github.com/delta-io/delta-rs) rust library.

Currently in delta-rs implementation we can use multiple storage backends including:
Local filesystem, [AWS S3](https://aws.amazon.com/s3/), 
[Azure Blob Storage](https://azure.microsoft.com/pl-pl/products/storage/blobs/) and [Azure Deltalake Storage Gen 2](https://learn.microsoft.com/en-us/azure/storage/blobs/data-lake-storage-introduction) and also [Google Cloud Storage](https://cloud.google.com/storage).

To be able to manage multiple delta tables on multiple stores I have built [Yummy delta server](https://www.yummyml.com/delta) which expose Rest API.

<img src="{{ site.relative_url }}assets/images/2023/03/YummyDelta.02.jpeg" alt="delta lake" width="900" />

Using API we can: list and create delta tables, inspect delta tables schema, append or override data in delta tables and additional operations like optimize or vacuum.

You can find API reference here: [https://www.yummyml.com/delta](https://www.yummyml.com/delta)

Moreover we can query data using [Data Fusion sql-s](https://arrow.apache.org/datafusion/).
Query results will be returned as a stream thus we can process it in batches.

You can simply install Yummy delta as a python package:
```bash
pip3 install yummy[delta]
```

Then we need to prepare config file:
```yaml
stores:
  - name: local
    path: "/tmp/delta-test-1/"
  - name: az
    path: "az://delta-test-1/"
```

And you are ready run server using command line:
```bash
yummy delta server -h 0.0.0.0 -p 8080 -f config.yaml
```
Now we are able to perform all operations using the REST API.

