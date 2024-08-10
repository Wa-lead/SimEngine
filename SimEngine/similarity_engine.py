from typing import List, Dict, Optional, Union, Callable
import numpy as np

from scipy.sparse import csr_matrix

from SimEngine.models.embedding import EmbeddingInterface
from SimEngine.preprocessing import Preprocessor
from SimEngine.metrics import METRICS
from SimEngine.models.ner import NERInterface
from SimEngine.utils import SimilarityDict

import logging
  

class SimilarityEngine:
    """
    A class for computing similarity scores between two lists of strings using various similarity metrics.
    """
    
    def __init__(self,
                 embedding_interface: EmbeddingInterface,
                 ner_interface: NERInterface = None,   
                 preprocessing: Optional[Union[List[Preprocessor], Preprocessor]] = [],
                 threshold: Optional[float] = 0.81,
                 top_k: Optional[int] = None
                 ):
        """
        Initializes the SimilarityEngine object.

        Args:
        - embedding_interface (EmbeddingInterface): An object that implements the EmbeddingInterface, which is used to embed the input strings.
        - ner_interface (NERInterface): An object that implements the NERInterface, which is used to extract named entities from the input strings.
        - preprocessing (Optional[Union[List[Preprocessor], Preprocessor]]): A preprocessor or list of preprocessors to apply to the input strings before computing similarity scores.
        - threshold (Optional[float]): The threshold above which two strings are considered similar.
        - top_k (Optional[int]): The maximum number of similar strings to return.
        """
        self.embedding_interface = embedding_interface
        self.ner_interface = ner_interface
        self.preprocessing = preprocessing if isinstance(preprocessing, list) else [preprocessing]
        self.threshold = threshold
        self.top_k = top_k

    def preprocess(self, data: List[str]) -> List[str]:
        """
        Applies the preprocessing steps to the input data.

        Args:
        - data (List[str]): A list of strings to preprocess.

        Returns:
        - List[str]: The preprocessed list of strings.
        """
        if isinstance(self.preprocessing, list):
            for preprocessor in self.preprocessing:
                data = preprocessor.transform(data)
        return data

    def bert_embedding(self, data: List[str]) -> np.ndarray:
        """
        Computes the BERT embeddings for the input data.

        Args:
        - data (List[str]): A list of strings to embed.

        Returns:
        - np.ndarray: The embeddings for the input data.
        """
        embeddings = []
        for sentence in data:
            if isinstance(self.embedding_interface.embedding_model, list):
                sentence = [model.embed(sentence).squeeze() for model in self.embedding_interface.embedding_model]
                sentence = np.concatenate(sentence, axis = 0)
                embeddings+=[sentence]
            else:
                embeddings+=[self.embedding_interface.embedding_model.embed(sentence).squeeze()]
                
        return np.array(embeddings)
    
    def ner_prediction(self, data: List[str]) -> List[List[str]]:
        """
        Extracts named entities from the input data.

        Args:
        - data (List[str]): A list of strings to extract named entities from.

        Returns:
        - List[List[str]]: A list of lists of named entities for each input string.
        """
        ner_tags = set()
        for sentence in data:
            if isinstance(self.ner_interface.ner_model, list):
                for model in self.ner_interface.ner_model:
                    entities = set(model.predict(sentence)[0])
                    ner_tags.update(entities)
            else:
                entities = set(self.ner_interface.ner_model.predict(sentence)[0])
                ner_tags.update(entities)
        return list(ner_tags)

    def _adjust_weights(self, weights: List[float]) -> List[float]:
        """
        Adjusts the weights for each similarity metric so that they sum to 1.

        Args:
        - weights (List[float]): A list of weights for each similarity metric.

        Returns:
        - List[float]: The adjusted weights.
        """
        logging.warning(f'Weights do not sum to 1. Adjusting weights to sum to 1.')
        weights = np.array(weights)
        weights = weights / weights.sum()
        return weights
    

    def _compute_similarity(self, x1: List[str], x2: List[str], model: Union[NERInterface, EmbeddingInterface]) -> csr_matrix:
        """
        Computes the similarity scores between two lists of strings using a given similarity metric.

        Args:
        - x1 (List[str]): The first list of strings.
        - x2 (List[str]): The second list of strings.
        - model (Union[Callable, NERInterface, EmbeddingInterface]): The similarity metric to use.

        Returns:
        - csr_matrix: The similarity scores between the two lists of strings.
        """
        if callable(model.similarity_metric):
            return model.similarity_metric(x1, x2)
        else:
            return METRICS[model.similarity_metric](x1, x2)
        
    
    def fit(self,
            x1: List[str],
            x2: List[str]
            ) -> SimilarityDict:
        """
        Computes the similarity scores between two lists of strings.

        Args:
        - x1 (List[str]): The first list of strings.
        - x2 (List[str]): The second list of strings.

        Returns:
        - SimilarityDict: A dictionary containing the similarity scores between the two lists of strings.
        """
        preprocessed_x1 = self.preprocess(x1)
        preprocessed_x2 = self.preprocess(x2)
        
        similarity_scores = []
        weights = [] # weights for each similarity metric
        if self.embedding_interface:
            vectorized_x1 = self.bert_embedding(preprocessed_x1)
            vectorized_x2 = self.bert_embedding(preprocessed_x2)
            similarity_scores.append(self._compute_similarity(vectorized_x1, vectorized_x2, self.embedding_interface))
            weights.append(self.embedding_interface.weight)
            
        if self.ner_interface:
            ner_tags1 = self.ner_prediction(preprocessed_x1)
            ner_tags2 = self.ner_prediction(preprocessed_x2)
            similarity_scores.append(self._compute_similarity(ner_tags1, ner_tags2, self.ner_interface))
            weights.append(self.ner_interface.weight)
        
        if sum(weights) != 1.0:
            weights = self._adjust_weights(weights)
            
        similarity_matrix = np.zeros_like(similarity_scores[0])
        for i, similarity_score in enumerate(similarity_scores):
            similarity_matrix += weights[i] * similarity_score
        
        return SimilarityDict(
                              x1=x1,
                              x2=x2,
                              similarity_matrix=similarity_matrix,
                              threshold=self.threshold,
                              top_k=self.top_k
                              )
    
    def predict(self, data: List[str]) -> Dict[str, float]:
        """
        Returns the similarity scores between the input data and the data used to fit the model.

        Args:
        - data (List[str]): The list of strings to compute similarity scores for.

        Returns:
        - Dict[str, float]: A dictionary containing the similarity scores between the input data and the data used to fit the model.
        """
        return self.similarity_dict[data]
