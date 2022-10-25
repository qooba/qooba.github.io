---
id: 737
title: 'Speedup features serving with Rust - Yummy serve'
date: '2022-10-25T11:00:00'
author: qooba
layout: post
guid: 'https://blog.qooba.net/?p=737'
permalink: /2022/10/25/speedup-features-serving-with-rust/
categories:
    - Feast
    - MLOps
    - 'feature store'
    - rust
tags:
    - Feast
    - 'feature store'
    - MLOps
    - rust
---

<img src="{{ site.relative_url }}assets/images/2022/10/snail-5352510_640.jpg" alt="slime" width="900" />

In this video I will introduce Yummy feature server implemented in Rust. 
The feature server is fully compatible with Feast implementation. 
Additionally benchmark results will be presented.

Before you will continue reading please watch short introduction: 

<div class="video-container">
    <iframe src="https://www.youtube.com/embed/lXCJLc3hWgY" frameborder="0" allowfullscreen></iframe>
</div>

Another step during MLOps process creation is features serving.

A historical feature store is used during model training to fetch a large range of entities
and a large dataset with small numbers of queries. 
For this process the data fetch latency is important but not critical. 

On the other hand when we serve the model features, fetching latency is crucial and determines prediction time. 

<img src="{{ site.relative_url }}assets/images/2022/10/YummyServe.01.jpeg" alt="feature store" width="900" />

That’s why we use very fast online stores like Redis or DynamoDb. 

<img src="{{ site.relative_url }}assets/images/2022/10/YummyServe.02.jpeg" alt="stores" width="900" />

The question which appears at this point is shall we call online store directly or use feature server ?

Because multiple models can reuse already prepared features 
we don't want to add feature store dependencies to the models. 
Thus we abstract an online store with a feature server which serves features
using for example REST api. 

<img src="{{ site.relative_url }}assets/images/2022/10/YummyServe.04.jpeg" alt="architecture" width="900" />

On the other hand latency due to additional layer should be minimized. 

Using Feast, we can manage features lifecycle
and we can serve features using built-in features server 
implemented in: python, java or go. 

<img src="{{ site.relative_url }}assets/images/2022/10/YummyServe.05.jpeg" alt="assumptions" width="900" />

According to the [provided benchmark](https://feast.dev/blog/feast-benchmarks/) Feast feature server is very fast. 
But can we go faster with the smaller number of computing resources ?

To answer this question I have implemented feature server using Rust 
which is known for its speed and safety. 

One of the basic assumptions was to ensure full compatibility 
with Feast and usage simplicity.
 
I have also decided to start implementation
with Redis as an online store.

The whole [benchmark code](https://github.com/yummyml/feature-servers-benchmarks) is available on github.

To reproduce benchmark we will clone the repository:
```bash
git clone https://github.com/yummyml/feature-servers-benchmarks.git
cd feature-servers-benchmarks
```

For simplicity I will use docker. 
Thus in the first step we will prepare all required 
images: Feast and Yummy feature server, Vegeta attack load generator 
and Redis.

```bash
./build.sh
```

Then I will use data generator to prepare dataset
apply feature store and materialize it to Redis. 

```bash
./materialize.sh
```

Now we are ready to start the benchmark. 

In contrast to the Feast benchmark where they used 
sixteen feature store server instances I will perform 
it with a single instance to simulate behavior 
on the smaller number of resources. 

The whole benchmark contains multiple scenarios like 
changing number of entities, number of features or increasing
number of requests per second.

```bash
# single_run <entities> <features> <concurrency> <rps>

echo "Change only number of rows"

single_run 1 50 $CONCURRENCY 10

for i in $(seq 10 10 100); do single_run $i 50 $CONCURRENCY 10; done


echo "Change only number of features"

for i in $(seq 50 50 250); do single_run 1 $i $CONCURRENCY 10; done


echo "Change only number of requests"

for i in $(seq 10 10 100); do single_run 1 50 $CONCURRENCY $i; done

for i in $(seq 100 100 1000); do single_run 1 50 $CONCURRENCY $i; done

for i in $(seq 10 10 100); do single_run 1 250 $CONCURRENCY $i; done

for i in $(seq 10 10 100); do single_run 100 50 $CONCURRENCY $i; done

for i in $(seq 10 10 100); do single_run 100 250 $CONCURRENCY $i; done
```

All results are available on GitHub but here I will limit it to `p99`
response time analysis for different numbers of requests. 

All results were performed on my local machine 
with 6 cpu cores 2.59 GHz and 32 GB of memory.

During these tests I will fetch a single entity 
with fifty features using feature service. 

To run Rust feature server benchmark we will run:
```bash
./run_test_yummy.sh
```

For Rust implementation `p99` response times are stable and less 
than 4 ms going from 10 requests per seconds to 100 requests per second.

<img src="{{ site.relative_url }}assets/images/2022/10/YummyBenchmark.png" alt="yummy benchmark results" width="900" />

For Feast following [documentation](https://docs.feast.dev/reference/feature-servers/go-feature-server) 
I have set `go_feature_retrieval` to `True` 
in `feature_store.yaml`

```yaml
registry: registry.db
project: feature_repo
provider: local
online_store:
  type: redis
  connection_string: redis:6379
offline_store:
  type: file
go_feature_retrieval: True
entity_key_serialization_version: 2
```

Additionally `go` option in feast serve command line. 
```bash
feast serve --host "0.0.0.0" --port 6566 --no-access-log --no-feature-log --go
```

Thus I assume that go implementation of the feature server will be used. 
In this part I have used the official `feastdev/feature-server:0.26.0` Feast docker image. 

Again I will fetch a single entity with fifty features using feature service. 
For 10 requests per second the p99 response time is 92 ms.

Unfortunately for 20 requests per seconds and above the p99 response 
time is above 5s which exceeds our timeout value.  

<img src="{{ site.relative_url }}assets/images/2022/10/FeastBenchmark.png" alt="feast benchmark results" width="900" />

Additionally during Feast benchmark run I have noticed increasing 
memory allocation which can be caused by the memory leak. 

This benchmark indicates that rust implementation is very promising 
because response times are small and stable, 
additionally the resources consumption is low.

The Yummy feature server usage description is available on:
[https://www.yummyml.com/feature-server](https://www.yummyml.com/feature-server)


