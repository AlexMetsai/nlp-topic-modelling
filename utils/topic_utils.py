"""
Alexandros Metsai
alexmetsai@gmail.com

Utils for topic modelling.
"""

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def tokenize_review(article, extra_stops=None):
    # common stop words in the English language
    en_stopwords = stopwords.words('english')

    if extra_stops:
        en_stopwords += extra_stops

    # stop words collection as a fast searchable set
    review_stopwords = set(en_stopwords)

    lmr = WordNetLemmatizer()

    # tokenize the text
    review_tokens = []
    for t in word_tokenize(article):
        if t.isalpha():
            t = lmr.lemmatize(t.lower())
            if t not in review_stopwords:
                review_tokens.append(t)

    return review_tokens
