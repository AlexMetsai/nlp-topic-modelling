"""
Alexandros Metsai
alexmetsai@gmail.com

Various utilities used for our analysis.
"""

import pandas as pd


def match_url_to_company(df):
    """
    Find which company each review concerns, based on the url that it got submitted to.

    Using lambda and pandas' apply() seems like the most efficient way to do this for now (since
    iterating on a Dataframe should be avoided if possible).
    """

    # Remove url prefixes
    prefixes = ['https://uk.trustpilot.com/review/www.',
                'https://uk.trustpilot.com/review/']
    for prefix in prefixes:
        df['url'] = df['url'].apply(lambda x: x.replace(prefix, ''))

    # Skip everything after the first dot ('.').
    df['url'] = df['url'].apply(lambda x: x.split('.')[0])

    return df.rename(columns={'url': 'company'})


def load_and_preprocess_reviews(path='train_reviews.json'):
    """
    :param path: Path to the json file.
    :return df: A preprocessed pandas dataframe.
    """

    df = pd.read_json(path)

    # Format reviews column to contain only a single int, the stars/rating.
    df['stars'] = df['stars'].apply(lambda x: int(x[24]))

    # Find which company each review concerns, based on the url that it got submitted to.
    df = match_url_to_company(df)

    return df


if __name__ == '__main__':
    pass
