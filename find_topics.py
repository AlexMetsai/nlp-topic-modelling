"""
Alexandros Metsai
alexmetsai@gmail.com

Find topics discussed in the the Trustpilot dataset, using NLTK and gensim.
Got useful insight from this article:
    https://thecodinginterface.com/blog/nlp-topic-modeling/
"""

import matplotlib.pyplot as plt

from gensim.models.ldamodel import LdaModel
from utils.utils import load_and_preprocess_reviews
from utils.topic_utils import tokenize_review
from gensim.corpora.dictionary import Dictionary


def find_topics(texts, n_topics=8):

    # Tokenize the text
    docs = []
    for txt in texts:
        # for some reason, 'was'->'wa', omit for now.
        texts_tokens = tokenize_review(txt, extra_stops=['wa', 'train'])
        docs.append(texts_tokens)

    # Create a dictionary and use the bag-of-words format.
    corpus_dict = Dictionary(docs)
    corpus = [corpus_dict.doc2bow(doc) for doc in docs]

    # Train an unsupervised LDA model of with n_topics.
    lda = LdaModel(corpus, num_topics=n_topics, random_state=23, id2word=corpus_dict)

    # Get the topics.
    topics = lda.show_topics(num_topics=n_topics, num_words=5, formatted=False)
    topics = sorted(topics, key=lambda x: int(x[0]))

    # Plot the topics
    rows = 4
    cols = 2
    fig, axs = plt.subplots(nrows=rows, ncols=cols, sharex=True, figsize=(12, 8))
    for topic_id, word_props in topics:
        row = topic_id // cols
        col = topic_id - (row * cols)

        ax = axs[row, col]
        words, probs = zip(*word_props)
        ax.barh(words, probs)
        ax.invert_yaxis()
        ax.set_title('Topic {}'.format(topic_id))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # Extract topics for all dates
    reviews = load_and_preprocess_reviews()
    reviews_text = reviews['text'].values
    find_topics(reviews_text)

    # Find topics over time, every 6 months.
    half_year = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
    for year in [2015, 2016, 2017, 2018]:
        for months in half_year:
            reviews_m = reviews[reviews['date'].dt.year == year]
            reviews_m = reviews_m[reviews_m['date'].dt.month.isin(months)]
            reviews_text_m = reviews_m['text'].values
            if reviews_text_m.size == 0:
                # no data, continue
                print(f"No data found for {year}-{months}")
                continue
            print(f"Topics for {year} and mothns: {months}")
            find_topics(reviews_text_m)

    # Find topics over time, in a monthly manner.
    for year in [2015, 2016, 2017, 2018]:
        for month in range(1, 13):
            reviews_m = reviews[reviews['date'].dt.year == year]
            reviews_m = reviews_m[reviews_m['date'].dt.month == month]
            #reviews_text_m = [''.join(x) for x in reviews_m['text']]
            reviews_text_m = reviews_m['text'].values
            if reviews_text_m.size == 0:
                # no data, continue
                print(f"No data found for {year}-{month}")
                continue
            print(f"Topics for {year}-{month}")
            find_topics(reviews_text_m)
