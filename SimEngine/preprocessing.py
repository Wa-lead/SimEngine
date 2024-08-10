from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from numpy.typing import NDArray

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from arabert.preprocess import ArabertPreprocessor as arabert_preprocessor


class Preprocessor(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def transform(self, data: NDArray) -> NDArray:
        pass

class HardPreprocessor(Preprocessor):
    
    WORDS_TO_EXCLUDE: List[str] =  [
        "عهده",
        "عهد",
        "عهدة",
        "عهدت",
        "سلفة",
        "سلف",
        "سلفه",
        "سلفت",
        "مستديمة",
        "مستديمه",
        "مستديم",
        "مؤقته",
        "مؤقتة",
        "مؤقت",
        "مؤقتين",
    ]
    def __init__(self, words_to_exclude: List[str] = None):
        super().__init__()
        if words_to_exclude:
            self.WORDS_TO_EXCLUDE += words_to_exclude
        
    def transform(self, data: NDArray) -> NDArray:
        for i in range(len(data)):
            for unwanted_word in self.WORDS_TO_EXCLUDE:
                data[i] = data[i].replace(unwanted_word, '')
        return data   
        
    
class TFIDFPreprocessor(Preprocessor):
    def __init__(self):
        super().__init__()
        self.vectorizer = TfidfVectorizer()
        
    def transform(self, data: NDArray) -> NDArray:
        self._fit(data)
        idf = self.get_idf()
        threshold = self._dynamic_thresholding()
        words_to_keep = {word for word, weight in idf.items() if weight >= threshold}
        preprocessed_data = []
        for sentence in data:
            preprocessed_data.append(' '.join([word for word in sentence.split() if word in words_to_keep]))
        return preprocessed_data
    
    def get_tf(self) -> Dict:
        return self.vectorizer.vocabulary_
    
    def get_idf(self) -> Dict:
        idf = self.vectorizer.idf_
        return dict(zip(self.vectorizer.get_feature_names_out(), idf))
    
    def _dynamic_thresholding(self) -> float:
        idf = list(self.get_idf().values())
        idf_mean = np.mean(idf)
        idf_std = np.std(idf)
        bottom_outliers = idf_mean - 2 * idf_std
        return bottom_outliers
    
    def _fit(self, data: NDArray) -> None:
        self.vectorizer.fit(data)
    

class ArabertPreprocessor(Preprocessor):
    def __init__(self):
        super().__init__()
        self.arabert_preprocessor = arabert_preprocessor("bert-base-arabertv02")
        
    def transform(self, data: NDArray) -> NDArray:
        transformed_data = self.arabert_preprocessor.preprocess(data)
        return list(transformed_data)
    
  
if __name__ == '__main__':
    
    print('This is a module for preprocessing.')
    docs = ['This is the first document.', 'This is the second second document.', 'And the the the the third one.', 'Is this the first document?']
    print(np.char.replace(docs, 't', ' '))