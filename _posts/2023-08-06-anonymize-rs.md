---
id: 742
title: 'Data anonymization with AI'
date: '2023-08-06T08:00:00'
author: qooba
layout: post
guid: 'https://blog.qooba.net/?p=742'
permalink: /2023/08/06/data-anonymization-with-ai/
categories:
    - Anonymization
    - Rust
    - AI
    - MachineLearning
    - LanguageModels
    - ArtificialIntelligence
tags:
    - ArtificialIntelligence
    - Rust
    - AI
    - MachineLearning
    - LanguageModels
    - Anonymization
    - Security
---

<img src="{{ site.relative_url }}assets/images/2023/08/masks-1559215_640.jpg" alt="all" width="900" />

Data anonymization is the process of protecting private or sensitive information 
by erasing or encrypting identifiers that link an individual to stored data. 
This method is often used in situations where privacy is necessary, 
such as when sharing data or making it publicly available. 
The goal of data anonymization is to make it impossible (or at least very difficult) to 
identify individuals from the data, while still allowing the data to be useful for analysis and research purposes.

Before you will continue reading please watch short introduction: 

<div class="video-container">
    <iframe src="https://www.youtube.com/embed/wJImdn0UvXs" frameborder="0" allowfullscreen></iframe>
</div>

I have decided to create a library which will help to simply anonymize data with high-performance.
That's why I have used Rust to code it. 
The library will use three algorithms which will anonymize data.
Named Entity Recognition method enables the library to identify 
and anonymize sensitive named entities in your data, 
like names, organizations, locations, and other personal identifiers.

Here you can use existing models from HuggingFace for different languages for example:
* [dslim bert-base-NER](https://huggingface.co/dslim/bert-base-NER) for english
* [clarin FastPDN](https://huggingface.co/clarin-pl/FastPDN) for polish 

The models are based on external libraries like pytorch. To avoid external dependencies 
I have used rust tract library which is a rust onnx implementation. 

To use models we need to convert them to onnx format using the transformers library. 

```python
import os
import transformers
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForTokenClassification
from transformers.onnx import FeaturesManager
from pathlib import Path
from transformers import pipeline

model_id='dslim/bert-base-NER'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForTokenClassification.from_pretrained(model_id)

feature='token-classification'

model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
onnx_config = model_onnx_config(model.config)

output_dir = "./dslim"
os.makedirs(output_dir, exist_ok=True)

# export
onnx_inputs, onnx_outputs = transformers.onnx.export(
        preprocessor=tokenizer,
        model=model,
        config=onnx_config,
        opset=13,
        output=Path(output_dir+"/model.onnx")
)

print(onnx_inputs)
print(onnx_outputs)
tokenizer.save_pretrained(output_dir)
```

Now we are ready to use the NER algorithm. 
We can simply run docker images with a yaml configuration file where we define an anonymization pipeline.

```bash
pipeline:
  - kind: ner
    model_path: ./dslim/model.onnx
    tokenizer_path: ./dslim/tokenizer.json
    token_type_ids_included: true
    id2label:
      "0": ["O", false]
      "1": ["B-MISC", true]
      "2": ["I-MISC", true]
      "3": ["B-PER", true]
      "4": ["I-PER", true]
      "5": ["B-ORG", true]
      "6": ["I-ORG", true]
      "7": ["B-LOC", true]
      "8": ["I-LOC", true]
```

```bash
docker run -it -v $(pwd):/app/ -p 8080:8080 qooba/anonymize-rs server --host 0.0.0.0 --port 8080 --config config.yaml
```

For the NER algorithm we can configure if the predicted entity will be replaced or not.
For the example request we will receive an anonymized response and replace items. 

```bash
curl -X GET "http://localhost:8080/api/anonymize?text=I like to eat apples and bananas and plums" -H "accept: application/json" -H "Content-Type: application/json"
```

Response:
```json
{
    "text": "I like to eat FRUIT_FLASH0 and FRUIT_FLASH1 and FRUIT_REGEX0",
    "items": {
        "FRUIT_FLASH0": "apples",
        "FRUIT_FLASH1": "banans",
        "FRUIT_REGEX0": "plums"
    }
}
```

If needed we can deanonymize the data using a separate endpoint. 
```bash
curl -X POST "http://localhost:8080/api/deanonymize" -H "accept: application/json" -H "Content-Type: application/json" -d '{
    "text": "I like to eat FRUIT_FLASH0 and FRUIT_FLASH1 and FRUIT_REGEX0",
    "items": {
        "FRUIT_FLASH0": "apples",
        "FRUIT_FLASH1": "banans",
        "FRUIT_REGEX0": "plums"
    }
}'
```

Response:
```json
{
    "text": "I like to eat apples and bananas and plums"
}
```


If we prefer we can use the library from python code in this case we simply install the library.
And we can use it in python. 

We have discussed the first anonymization algorithm but what if it is not enough ? 
There are two additional methods. First is [Flush Text](https://arxiv.org/abs/1711.00046) algorithm which is
a fast method for searching and replacing words in large datasets, 
used to anonymize predefined sensitive information. 
For flush text we can define configuration where we can read keywords in separate file
where each line is a keyword or in the keyword configuration section. 

The last method is simple Regex where we can define patterns which will be replaced.

We can combine several methods and build an anonymization pipeline:

```yaml
pipeline:
  - kind: ner
    model_path: ./dslim/model.onnx
    tokenizer_path: ./dslim/tokenizer.json
    token_type_ids_included: true
    id2label:
      "0": ["O", false]
      "1": ["B-MISC", true]
      "2": ["I-MISC", true]
      "3": ["B-PER", true]
      "4": ["I-PER", true]
      "5": ["B-ORG", true]
      "6": ["I-ORG", true]
      "7": ["B-LOC", true]
      "8": ["I-LOC", true]
  - kind: flashText
    name: FRUIT_FLASH
    file: ./tests/config/fruits.txt
    keywords:
    - apple
    - banana
    - plum
  - kind: regex
    name: FRUIT_REGEX
    file: ./tests/config/fruits_regex.txt
    patterns:
    - \bapple\w*\b
    - \bbanana\w*\b
    - \bplum\w*\b
```

Remember that it 
uses automated detection mechanisms, and there is no guarantee that it will find all sensitive information. 
You should always ensure that your data protection measures are comprehensive and multi-layered.