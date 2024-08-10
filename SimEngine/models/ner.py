from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Union, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

class NERModel(ABC):
    """
    Abstract base class for Named Entity Recognition models.
    """

    def __init__(self, model_name: str):
        """
        Initializes the NERModel object.

        Args:
        - model_name (str): The name of the pre-trained model to use.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name)

    @abstractmethod
    def predict(self, data: NDArray):
        """
        Abstract method for predicting named entities in the input data.

        Args:
        - data (NDArray): The input data to predict named entities for.

        Returns:
        - A list of named entities for each sentence in the input data.
        """
        pass


class Hatmimoha(NERModel):
    """
    Named Entity Recognition model for Arabic text using the Hatmimoha pre-trained model.
    """

    def __init__(self):
        """
        Initializes the Hatmimoha object.
        """
        super().__init__("hatmimoha/arabic-ner")

    def extract_tags_from_pairs(self, token_label_pairs: list):
        """
        Extracts tags from a list of token-label pairs.

        Args:
        - token_label_pairs (list): A list of token-label pairs.

        Returns:
        - A dictionary mapping tokens to their corresponding labels.
        """
        # Remove special tokens (like [CLS], [SEP], etc.)
        token_label_pairs = [
            pair for pair in token_label_pairs if pair[0] not in self.tokenizer.all_special_tokens]

        # If we use WordPiece tokenizer, some words might be split, so we should merge them back
        merged_labels = {}
        current_word = ""
        current_label = None
        for token, label in token_label_pairs:
            if token.startswith("##"):
                current_word += token[2:]
            else:
                if current_word:
                    merged_labels[current_word] = current_label
                    current_word = ""
                current_word = token
                current_label = label
        if current_word:
            merged_labels[current_word] = current_label

        return merged_labels

    def predict(self, data: NDArray):
        """
        Predicts named entities in the input data.

        Args:
        - data (NDArray): The input data to predict named entities for.

        Returns:
        - A list of named entities for each sentence in the input data.
        """
        # Guard against single string input
        if isinstance(data, str):
            data = [data]

        inputs = self.tokenizer(data, return_tensors="pt", padding=True)
        outputs = self.model(**inputs).logits
        predictions = np.argmax(outputs.detach().numpy(), axis=2)

        # Extract tags for each sentence
        sentence_entities = []
        for i, word in enumerate(data):
            token_label_dict = self.extract_tags_from_pairs(list(zip(self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].numpy()[i]),
                                                                     [self.model.config.id2label[pred] for pred in predictions[i]])))
            sentence_entities.append(
                self._extract_entities_single(token_label_dict))

        return sentence_entities

    def _extract_entities(self, output: Dict[str, str]) -> List[str]:
        """
        Extracts named entities from the output of the model.

        Args:
        - output (Dict[str, str]): The output of the model.

        Returns:
        - A list of named entities.
        """
        entities = []
        current_entity = []

        for word, tag in output.items():
            if tag != 'O':  # If the is not an outside tag
                current_entity.append(word)
            else:
                if current_entity:  # save the entity if it exists
                    entities.append(" ".join(current_entity))
                    current_entity = []

        if current_entity:  # saving any remaining entity
            entities.append(" ".join(set(entities)))

        return entities

    def _extract_entities_single(self, output):
        """
        Extracts named entities from the output of the model for a single sentence.

        Args:
        - output (Dict[str, str]): The output of the model for a single sentence.

        Returns:
        - A list of named entities.
        """
        entities = []

        for word, tag in output.items():
            if tag != 'O':
                entities.append(word)

        return entities


@dataclass
class NERInterface():
    def __init__(self, ner_model: Union[NERModel, List[NERModel]] = Hatmimoha(),
                 similarity_metric: Union[str, callable] = 'jaccard_edit',
                 weight: Optional[float] = 1,
                 ):
        self.ner_model = ner_model
        self.similarity_metric = similarity_metric
        self.weight = weight
        


if __name__ == '__main__':
    model = Hatmimoha()
    data1 = ['أنا أحب القطط', 'انا اعيش في الرياض']
    data2 = ['أنا أحب القطط', 'انا اعيش في الرياض']
    from SimEngine.metrics import jaccard_similarity_with_edit_distance
    print(jaccard_similarity_with_edit_distance(model.predict(data1), model.predict(data2)))
