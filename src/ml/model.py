from gensim.models.fasttext import FastText
from gensim.utils import simple_preprocess
from typing import List, Tuple
import os
from data_collection import Location
import numpy as np
from numpy import dot
from numpy.linalg import norm
from nltk.corpus import stopwords
# import nltk
# nltk.download('stopwords')

class Model:
    def __init__(self):
        self.model = FastText.load(os.path.join(os.path.abspath(os.path.dirname(__file__)), "models/fasttext"))
        self.stopwords = set(stopwords.words('english'))

    def _cosine_similarity(self, text1: str, text2: str)-> float:
        embed1 = np.mean([self.model.wv[token] for token in simple_preprocess(text1) if token not in self.stopwords], axis=0)
        embed2 = np.mean([self.model.wv[token] for token in simple_preprocess(text2) if token not in self.stopwords], axis=0)
        # Cosine similarity
        return dot(embed1, embed2)/(norm(embed1))/(norm(embed2))

    def top(self, keyword: str, locations: List[Location]) \
            -> List[Tuple[Location, float]]:
        def get_similarity(word: str, location: Location) -> float:
            information = location.name + ' '.join(location.information)
            return self._cosine_similarity(word, information)

        result = []
        for loc in locations:
            result.append((loc, get_similarity(keyword, loc)))

        result.sort(key=lambda x: x[1])
        return result

    def compute_score(self, keyword: str, compare_to: str) -> float:
        return self._cosine_similarity(keyword, compare_to)