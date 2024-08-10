from transformers import AutoTokenizer, AutoModel
import fasttext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

from nltk.tokenize import word_tokenize
import numpy as np

from numpy.typing import NDArray
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod


class Embedder(ABC):
    """Abstract base class for all embedders."""

    @abstractmethod
    def embed(self, data: NDArray) -> NDArray:
        pass


class TransformerEmbedder(Embedder):
    def __init__(self, tokenizer: str, model: str, pooling_strategy: str = 'mean'):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model = AutoModel.from_pretrained(model)
        self.pooling_strategy = pooling_strategy

    def tokenize(self, data: NDArray) -> Dict[str, NDArray]:
        return self.tokenizer(data, return_tensors="pt", padding=True, truncation=True, max_length=512)

    def embed(self, data: NDArray) -> NDArray:
        outputs = self.model(**self.tokenize(data))
        if self.pooling_strategy == 'cls':
            return outputs.last_hidden_state[:, 0].detach().numpy()
        elif self.pooling_strategy == 'mean':
            return outputs.last_hidden_state.mean(dim=1).detach().numpy()
        elif self.pooling_strategy == 'max':
            return outputs.last_hidden_state.max(dim=1).values.detach().numpy()
        elif self.pooling_strategy == 'pooler':
            return outputs.pooler_output.detach().numpy()
        else:
            raise ValueError(f'Pooling algorithm {self.pooling_strategy} is not supported')


class TFIDFEmbedder(Embedder):
    def __init__(self, tfidf: TfidfVectorizer = None, pca: Optional[PCA] = None):
        self.tfidf_vectorizer = tfidf if tfidf else TfidfVectorizer()
        self.pca = pca
        
    def embed(self, data: NDArray) -> NDArray:
        # check if the if the vecotrizer is already fitted
        if not self.tfidf_vectorizer.vocabulary_:
            self.tfidf_vectorizer.fit(data)
            
        vectorized_data = self.tfidf_vectorizer.transform(data)
        
        if self.pca:
            return self.pca.transform(vectorized_data.toarray())
        else:
            return vectorized_data.toarray()


class FastTextEmbedder(Embedder):
    def __init__(self, model_path: str,
                 word_weights: Optional[Dict[str, float]] = None,
                 pooling_strategy: str = 'mean'
                 ):
        """
        Initializes a FastTextEmbedder object.

        Args:
            model_path (str): The path to the FastText model file.
            word_weights (Optional[Dict[str, float]], optional): A dictionary of word weights. Defaults to None.
            pooling_strategy (str, optional): The pooling strategy to use. Can be 'mean' or 'max'. Defaults to 'mean'.
        """
        self.model = fasttext.load_model(model_path)
        self._word_weights = word_weights
        self.pooling_strategy = pooling_strategy
        
    def embed(self, data: NDArray) -> NDArray:
        """
        Embeds the given data using the FastText model.

        Args:
            data (NDArray): The data to embed.

        Returns:
            NDArray: The embedded data.
        """
        tokens = word_tokenize(data)
        token_embeddings = []

        for token in tokens:
            weight = self._word_weights.get(token, 1.0)  # default weight if not found in dictionary
            if token in self.model:
                token_embedding = self.model[token] * weight
                token_embeddings.append(token_embedding)

        if token_embeddings:
            if self.pooling_strategy == 'mean':
                sentence_embedding = np.mean(token_embeddings, axis=0)
            elif self.pooling_strategy == 'max':
                sentence_embedding = np.max(token_embeddings, axis=0)
            else:
                raise ValueError(f'Pooling algorithm {self.pooling_strategy} is not supported')
        else:
            sentence_embedding = np.zeros(self.model.get_dimension())

        return np.array(sentence_embedding)



class CAMeL(TransformerEmbedder):
    def __init__(self,
                tokenizer: str = 'CAMeL-Lab/bert-base-arabic-camelbert-msa',
                model: str = 'CAMeL-Lab/bert-base-arabic-camelbert-msa',
                pooling_strategy: str = 'mean'):
        
        super().__init__(tokenizer=tokenizer, model=model, pooling_strategy=pooling_strategy)


class AraBERTv2(TransformerEmbedder):
    def __init__(self, 
                 tokenizer: str = 'aubmindlab/bert-base-arabertv2',
                 model: str = 'aubmindlab/bert-base-arabertv2',
                 pooling_strategy: str = 'mean'
                 ):
        super().__init__(tokenizer=tokenizer, model=model, pooling_strategy=pooling_strategy)
                 


class MARBERT(TransformerEmbedder):
    def __init__(self,
                 tokenizer: str = 'aubmindlab/bert-base-arabertv2',
                 model: str = 'fasttext_models/cc.ar.300.bin',
                 pooling_strategy: str = 'mean'
                 ):
        super().__init__(tokenizer=tokenizer, model=model, pooling_strategy=pooling_strategy)


    
class FastTextArabicEmbedder(FastTextEmbedder):
    def __init__(self, 
                 model_path: str = '/Users/waleedalasad/Documents/GitHub/SimEngine/models/fasttext_models/cc.ar.300.bin',
                 word_weights: Optional[TfidfVectorizer] = None,
                 pooling_strategy: str = 'mean'
                 ):
        super().__init__(model_path=model_path, word_weights=word_weights, pooling_strategy=pooling_strategy)



@dataclass
class EmbeddingInterface():
     def __init__(self, embedding_model: Union[List[Embedder], Embedder] = CAMeL(),
                        similarity_metric: Union[str, callable] = 'cosine',
                        weight: Optional[float] = 1
                        ):
        self.embedding_model = embedding_model
        self.similarity_metric = similarity_metric
        self.weight = weight
        
        
    