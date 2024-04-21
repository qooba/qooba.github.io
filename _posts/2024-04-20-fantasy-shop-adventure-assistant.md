---
id: 746
title: 'Fantasy Shop 🐉⚔️ RAG Assistant 🛠️ crafted with Gemma and Rust 🦀'
date: '2024-04-20T08:00:00'
author: qooba
layout: post
guid: 'https://blog.qooba.net/?p=745'
permalink: /2024/04/20/fantasy-shop-rag-assistant-crafted-with-gemma-and-rust/
categories:
    - textembeddings
    - LLM
    - AI
    - MachineLearning
    - LanguageModels
    - ArtificialIntelligence
    - WebAssembly
    - Rust
    - RAG
    - VectorDatabase
tags:
    - ArtificialIntelligence
    - LLM
    - AI
    - MachineLearning
    - LanguageModels
    - TextEmbeddings
    - LanceDb
    - Polars
    - Delta-rs
    - Gemma
    - Mistral7b
---

<img src="{{ site.relative_url }}assets/images/2024/04/bubble.png" alt="bubble" width="900" />

Today's goal is to build an assistant for heroes who need to choose appropriate weapons for their adventures.

Before you will continue reading please watch short introduction: 

<div class="video-container">
    <iframe src="https://www.youtube.com/embed/NuUI4NfnxoY" frameborder="0" allowfullscreen></iframe>
</div>

To develop our RAG-s solution, we will go through several steps: collecting and preparing a dataset, calculating embeddings, choosing an appropriate vector database, and finally, using an open-source large language model to build an assistant.

<img src="{{ site.relative_url }}assets/images/2024/04/path.png" alt="path" width="900" />

In the first step, we will collect a dataset, our dataset will be in Delta Lake format. To read it, we will use two Python packages that are built with Rust under the hood: Polars, which is a blazing-fast dataframe package, and delta-rs, which simplifies reading Delta tables without Spark.

[FantasyAssistantPrep.ipynb](https://github.com/onceuponai-dev/stories-fantasy-shop-assistant/blob/main/notebooks/FantasyAssistantPrep.ipynb)

```python
import polars as pl

table_path="../data/fantasy-library"
df=pl.read_delta(table_path)
df = df.with_columns(('For '+pl.col('Scenario')+' you should use '+pl.col('Advice')).alias('Combined'))
print(df)
```

To read a Delta table, we can simply use the read_delta method. Our delta contains two columns: Scenario and Advice. We will create an additional context column called Combined, which is simply a concatenation of the Scenario and Advice columns.

Now it's time to calculate embeddings, which are multidimensional vectors calculated, for example, from text. To do this, we will use the E5 small model together with the Candle library.

Now it's time to write some code in Rust. We will use the candle_transformers library to create an E5Model struct and add two methods. The first will download the model from Hugging Face, and the second will calculate embeddings for provided texts.

[adventures/src/lib.rs](https://github.com/onceuponai-dev/stories-fantasy-shop-assistant/blob/main/adventures/src/lib.rs#L85)
``` rust
pub struct E5Model {
    pub model: BertModel,
    pub tokenizer: Tokenizer,
    pub normalize_embeddings: Option<bool>,
}

impl E5Model {
    pub fn load() -> Result<E5Model> {
        ...
    }

    pub fn forward(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>> {
        ...
    }
```

We would like to use our Rust code in Python; thus, we will use the additional PyO3 maturing packages. In our case, we will wrap our Rust code with the Python Adventures module and Adventures class. After compilation, we are ready to import our adventures module and calculate embeddings for our contexts.

[adventures-py/src/lib.rs](https://github.com/onceuponai-dev/stories-fantasy-shop-assistant/blob/main/adventures-py/src/lib.rs)
```rust
#[pymodule]
#[pyo3(name = "adventures")]
fn adventures(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyclass]
    pub struct Adventures {
        model: E5Model,
    }

    #[pymethods]
    impl Adventures {
        #[new]
        pub fn new() -> Self {
            let model = E5Model::load().unwrap();
            Self { model }
        }

        pub fn embeddings(&self, input: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
            let embeddings = self.model.forward(input).unwrap();
            Ok(embeddings)
        }
    }

    impl Default for Adventures {
        fn default() -> Self {
            Self::new()
        }
    }

    m.add_class::<Adventures>()?;
    Ok(())
}
```

[notebooks/FantasyAssistantPrep.ipynb](https://github.com/onceuponai-dev/stories-fantasy-shop-assistant/blob/main/notebooks/FantasyAssistantPrep.ipynb)
```python
import adventures

a = adventures.Adventures()

items = []
for combined_text in df['Combined']:
    emb = a.embeddings([combined_text])
    items.append({"item": combined_text, "vector": emb[0]})

items
```

Now it's time to choose a vector database where we will store our embeddings. To do this, we will use the LanceDb database. We can simply use the Python API to create a fantasy vectors table and create an index for it. 

```python
import lancedb
import numpy as np
uri = "/tmp/fantasy-lancedb"
db = lancedb.connect(uri)

tbl = db.create_table("fantasy_vectors", data=items)
tbl.create_index(num_partitions=256, num_sub_vectors=96)
```

Now we can confirm that we are able to use the created index to search for the most appropriate context. For example, in the first step, we calculate embeddings for "Adventure with a dragon" text. Then we search for the most appropriate context.

```python
import lancedb

emb = a.embeddings(["Adventure with a dragon"])
db = lancedb.connect("/tmp/fantasy-lancedb")
tbl = db.open_table("fantasy_vectors")
df = tbl.search(emb[0]) \
    .limit(1) \
    .to_list()


df[0]["_distance"]
```

It is time for the large language model. In our case, we will use the [Google Gemma model](https://huggingface.co/google/gemma-2b-it). Currently, Gemma models are published in two sizes: two billion and seven billion parameters. Additionally, we can use the instruct model type, which offers a specific turns prompt format that can be very helpful when building an assistant and wanting to keep the conversation context.

```
<start_of_turn>user
What is a good place for adventure ?<end_of_turn>
<start_of_turn>model
desert canyon.<end_of_turn>
<start_of_turn>user
What can I do in desert canyon ?<end_of_turn>
<start_of_turn>model
```

In our case, we will use the model with two billion parameters. Again, we will use the Rust Candle project to create a GemmaModel struct and a load method implementation. We aim to improve the user experience and, instead of creating a simple request-response method, we will use an additional async stream Rust library to stream text generated by the model.

[adventures/src/lib.rs](https://github.com/onceuponai-dev/stories-fantasy-shop-assistant/blob/main/adventures/src/lib.rs#L14)

```rust
pub struct GemmaModel {
    pub model: Model,
    pub device: Device,
    pub tokenizer: Tokenizer,
    pub logits_processor: LogitsProcessor,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

impl GemmaModel {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        base_repo_id: &str,
        model_endpoint: Option<String>,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        hf_token: Option<String>,
    ) -> Result<GemmaModel> {
        ...
    }
}
```

We have already collected the data, calculated embeddings, and indexed them into the LanceDb database. Now it's time to create a microservice that will expose a chat POST method where our heroes' team will send prompts. Inside the microservice, we will calculate embeddings using the E5 model, then search for the most appropriate context, build a large language model prompt in instruct format, and finally stream generated responses from the Gemma model to the heroes.

To build the microservice, we will use the [Actix web framework](https://actix.rs/).

During application start, we will load the Gemma model, E5 model, and additionally, we will create a LanceDb table object. For the Gemma model, we need to provide our Hugging Face token, which confirms that we have accepted the Gemma model license.

[adventures/src/main.rs](https://github.com/onceuponai-dev/stories-fantasy-shop-assistant/blob/main/adventures/src/main.rs#L315)
``` rust
static GEMMA_MODEL: OnceCell<Arc<Mutex<GemmaModel>>> = OnceCell::new();
 
//...
 
let model = GemmaModel::load(
    base_repo_id: GEMMA_2B_REPO_ID,
    model_endpoint: None,
    seed: 299792458,
    temp: Some(0.8),
    top_p: None,
    repeat_penalty: 1.1,
    repeat_last_n: 64,
    hf_token,
)
.unwrap();
 
GEMMA_MODEL.set(Arc::new(Mutex::new(model))).is_ok();
```

``` rust
static E5_MODEL: OnceCell<Arc<Mutex<E5Model>>> = OnceCell::new();
 
//...
 
let e5_model = E5Model::load().unwrap();
 
E5_MODEL.set(Arc::new(Mutex::new(e5_model))).is_ok();
```

``` rust
	static LANCEDB_TABLE: OnceCell<Arc<Mutex<lancedb::Table>>> = OnceCell::new();
 
//...
 
let uri = "/tmp/fantasy-lancedb";
let db = connect(uri).execute().await.unwrap();
let tbl = db.open_table("fantasy_vectors").execute().await.unwrap();
 
LANCEDB_TABLE.set(Arc::new(Mutex::new(tbl))).is_ok();
```

Inside the chat post method for request prompts, we will find the context which will cover calculating embeddings using the E5 model and searching for the most appropriate context.

[adventures/src/main.rs](https://github.com/onceuponai-dev/stories-fantasy-shop-assistant/blob/main/adventures/src/main.rs#L163)

``` rust
	pub async fn chat(
    request: web::Json<PromptRequest>,
    gemma_state: web::Data<GemmaState>,
) -> Result<impl Responder, Box<dyn Error>> {
    let context = find_context(request.prompt.to_string()).await.unwrap();
    let prompt = build_prompt(request.prompt.to_string(), context)
        .await
        .unwrap();
 
    let mut model = GEMMA_MODEL.get().unwrap().lock().await;
    model.model.clear_kv_cache();
 
    let mut tokens = model
        .tokenizer
        .encode(prompt.clone(), true)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();
 
    let stream_tasks = stream! {
	
        for index in 0..request.sample_len {
            // ...

            yield Ok::<Bytes, Box<dyn Error>>(byte);
        }
	
    };
 
    Ok(HttpResponse::Ok()
        .content_type("text/event-stream")
        .streaming(Box::pin(stream_tasks)))
}
```

Now we are ready to build an instruct prompt using a simple template.
Finally, we will pass the instruct prompt to the Gemma model and stream results.
In this case, we run the Gemma model on CPU..

Additionally to improve solution prerformance we will use a model quantization process, which reduces model weights precision.

In the next step, we will use another open-source large language model, Mistral, with seven billion parameters which use 4-bit quantization. We will use the Candle library to load the model in gguf format, but in this case, we will use CUDA to run it on a GPU card.

[adventures/src/lib.rs](https://github.com/onceuponai-dev/stories-fantasy-shop-assistant/blob/main/adventures/src/lib.rs#L152)

```rust
pub struct QuantizedModel {
    pub model: ModelWeights,
    pub tokenizer: Tokenizer,
    pub device: Device,
}

impl QuantizedModel {
    pub fn load() -> Result<QuantizedModel> {
        //let base_repo_id = ("TheBloke/CodeLlama-7B-GGUF", "codellama-7b.Q4_0.gguf");
        let base_repo_id = (
            "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            "mistral-7b-instruct-v0.2.Q4_K_S.gguf",
        );
        let tokenizer_repo = "mistralai/Mistral-7B-Instruct-v0.2";

        //...

        let device = Device::new_cuda(0).unwrap();

        //...
    }
}
```