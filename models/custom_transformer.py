import re
import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.base import BaseEstimator, TransformerMixin


class GetMessageLength(BaseEstimator, TransformerMixin):
    """Determine the number of words in a message."""

    def get_num_words(self, msg):
        """Create extra features for message
        1. how many words it has
        2. how many non-stopwords it has
        3. the ratio of non-stop to total words

        Args:
        msg {str}: a message

        Return:
        feats {np.array}: has the above mentioned 3 values
        """
        msg = re.sub(r"[^a-zA-Z0-9]", " ", msg.lower())
        # Split the message to list of words
        words = word_tokenize(msg)

        # Remove stop words, as thye are non-informative in our case
        stop_words = stopwords.words('english')
        tokens = [word for word in words if word not in stop_words]
        # To prevent `division by zero` I add one to the denominator
        feats = np.array([len(words), len(tokens), len(tokens) / (len(words) + 1)])
        return feats

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        res = np.array(list(map(self.get_num_words, X)))
        res = pd.DataFrame(res)
        res.columns = ['num_words', 'num_non_stops', 'ratio_info']
        return res
