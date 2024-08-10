# SimEngine

## Overview

The provided project computes similarity scores between two lists of strings using various similarity metrics. These metrics include cosine similarity, Jaccard similarity, and more.

## Project Structure

- `metrics.py`: This file contains multiple similarity and distance metrics.
  * Cosine Similarity
  * Jaccard Similarity
  * Jensen Shannon Divergence & Distance
  * Jaccard Similarity with Edit Distance

- `similarity_engine.py`: Central file containing the `SimilarityEngine` class which processes and computes similarity scores.

- `utils.py`: Utility functions including:
  * `SimilarityDict` dataclass 
  * Batch data generator
  * Functions to save results to Excel

- `preprocessing.py`: Handles data preprocessing with preprocessors such as:
  * HardPreprocessor
  * TFIDFPreprocessor
  * ArabertPreprocessor

## Getting Started

1. **Setup**:
   Ensure the installation of requirements.txt:
   ```bash
   pip install requirements.txt
   ```
   
3. **Usage**:
   
- 3.1. Quick Usage
```python
    from SimEngine.models.embedding import EmbeddingInterface
    from SimEngine.models.ner import NERInterface
    from SimEngine.similarity_engine import SimilarityEngine

    list1 = ["This is a sample string.", "Another example."]
    list2 = ["A different sample string.", "Yet another example."]
    
    # Initialize the embedding models
    embedding = EmbeddingInterface()
    
    # Initialize the NER models
    ner = NERInterface()   
    
    # Initialize the similarity engine
    engine = SimilarityEngine(embedding_interface=embedding, ner_interface=ner)
    
    # Fit the similarity engine
    sim_dict = engine.fit(x1 = list1, x2=list2)
```
  - 3.2. Detailed Usage
  ```python
  import pandas as pd
  import numpy as np
  from sklearn.feature_extraction.text import TfidfVectorizer
  
  from SimEngine.models.embedding import CAMeL, AraBERTv2, MARBERT, FastTextArabicEmbedder, TFIDFEmbedder, EmbeddingInterface
  from SimEngine.models.ner import Hatmimoha, NERInterface
  from SimEngine.preprocessing import TFIDFPreprocessor, ArabertPreprocessor, HardPreprocessor
  from SimEngine.similarity_engine import SimilarityEngine

  list1 = ["This is a sample string.", "Another example."]
  list2 = ["A different sample string.", "Yet another example."]

  # Prepare the word weights for FastTextArabicEmbedder
  tf_idf = TfidfVectorizer()
  tf_idf = tf_idf.fit(fc_text + contract_text)
  word_weights = dict(zip(tf_idf.get_feature_names_out(), tf_idf.idf_))
  
  # Initialize the embedding models
  embedding = EmbeddingInterface(
                                  embedding_model=[
                                                  CAMeL(pooling_strategy='mean'),
                                                  FastTextArabicEmbedder(word_weights = word_weights, pooling_strategy='max'),
                                                  ],
                                  similarity_metric = 'cosine',
                                  weight=0.85
  )
  
  # Initialize the NER models
  ner = NERInterface(
                  ner_model = Hatmimoha(),
                  weight = 0.15,
                  similarity_metric = 'jaccard_edit'
                  )   
  
  
  # Initialize the similarity engine
  engine = SimilarityEngine(
                            embedding_interface =  embedding, # Embedding models to use
                            ner_interface = ner, # NER models to use
                            preprocessing = [TFIDFPreprocessor()], # Preprocessing techniques to use
                            threshold = 0.80, # Min similarity score to consider
                            top_k = 10, # Return top k similar entires 
                            )
  
  sim_dict = engine.fit(x1 = fc_text, x2 = contract_text)
```
