# External libraries
import numpy as np
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
# Typing
from typing import Union, List, Dict, Callable
from numpy.typing import NDArray

# Levenshtein
import Levenshtein


def _softmax(x: Union[NDArray, List]):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def _normalize(x: Union[NDArray, List]):
    return x / np.linalg.norm(x)

def jensen_shannon_divergence(p: NDArray, q: NDArray):
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

def jensen_shannon_distance(p: NDArray, q: NDArray):
    return -1 * np.sqrt(jensen_shannon_divergence(p, q))

def jsd_similarity(A: NDArray, B:NDArray) -> NDArray:
    A,B = _softmax(A), _softmax(B)
    n = A.shape[0]
    similarity = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            similarity[i, j] = jensen_shannon_distance(A[i], B[j])   
    return similarity


# -- Embedding related functions -- #

def cosine_similarity(A: NDArray, B: NDArray) -> NDArray:
    return sklearn_cosine_similarity(A, B)
    

# -- NER related functions -- #
def jaccard_similarity(list1: Union[List[str], List[List[str]]], 
                       list2: Union[List[str], List[List[str]]]) -> Union[float, NDArray]:
    
    def basic_jaccard(l1: List[str], l2: List[str]) -> float:
        set1, set2 = set(l1), set(l2)
        intersection = len(set1.intersection(set2))
        union = len(set1) + len(set2) - intersection
        return float(intersection) / union if union != 0 else 0.0
    
    if isinstance(list1[0], list) and isinstance(list2[0], list):
        num_sentences1 = len(list1)
        num_sentences2 = len(list2)
        sim_matrix = np.zeros((num_sentences1, num_sentences2))

        for i, tags1 in enumerate(list1):
            for j, tags2 in enumerate(list2):
                sim_matrix[i][j] = basic_jaccard(tags1, tags2)

        return sim_matrix
    else:
        return basic_jaccard(list1, list2)
    

def jaccard_similarity_with_edit_distance(list1: Union[List[str], List[List[str]]], 
                                          list2: Union[List[str], List[List[str]]], 
                                          threshold: int = 2) -> Union[float, NDArray]:
    

    """
    Computes the Jaccard similarity between two lists of strings or lists of lists of strings, 
    using the Levenshtein distance to account for small differences between words. 
    If the input is two lists of lists of strings, it returns a matrix of similarities between 
    each pair of sentences. If the input is two lists of strings, it returns a single similarity value.
    
    Args:
    - list1: A list of strings or a list of lists of strings.
    - list2: A list of strings or a list of lists of strings.
    - threshold: An integer representing the maximum Levenshtein distance between two words to be 
                 considered a match. Default is 2.
    
    Returns:
    - If the input is two lists of lists of strings, returns a matrix of similarities between each 
      pair of sentences, as a numpy ndarray.
    - If the input is two lists of strings, returns a single similarity value, as a float.
    """
    
    def close_match(word: str, word_set: set, threshold: int = 2) -> bool:
        for potential_match in word_set:
            if Levenshtein.distance(word, potential_match) <= threshold:
                return True
        return False

    def jaccard_with_edit(l1: List[str], l2: List[str], threshold: int = 2) -> float:
        set1, set2 = set(l1), set(l2)
        intersection = sum(1 for word in set1 if close_match(word, set2, threshold))
        union = len(set1) + len(set2) - intersection
        return float(intersection) / union if union != 0 else 0.0
    
    if not list1 or not list2:
        return 0.0
    
    if isinstance(list1[0], list) and isinstance(list2[0], list):
        num_sentences1 = len(list1)
        num_sentences2 = len(list2)
        sim_matrix = np.zeros((num_sentences1, num_sentences2))

        for i, tags1 in enumerate(list1):
            for j, tags2 in enumerate(list2):
                sim_matrix[i][j] = jaccard_with_edit(tags1, tags2, threshold)

        return sim_matrix
    else:
        return jaccard_with_edit(list1, list2, threshold)


# -- Similarity metrics -- #

METRICS: Dict[str, Callable] = {
    'cosine': cosine_similarity,
    'jaccard': jaccard_similarity,
    'jaccard_edit': jaccard_similarity_with_edit_distance,
    'jsd': jsd_similarity,
}

if __name__ == "__main__":
    # Example usage:
    list1 = ['Riyadh', 'Waleed']
    list2 = ["New York City", "Los Angeles"]
    print(jaccard_similarity_with_edit_distance(list1, list2))


