"""
Alexandros Metsai
alexmetsai@gmail.com
"""


import pandas as pd


def match_url_to_company(df):
    """
    Find which company each review concerns, based on the url that it got submitted to.
    The list of companies was acquired through manual inspection, which translates to printing the unique urls :-)

    :param df:
    :return processed_df:
    """
    prefixes = ['https://uk.trustpilot.com/review/www.',
              'https://uk.trustpilot.com/review/']

    # Remove url prefixes
    for prefix in prefixes:
        df['url'] = df['url'].apply(lambda x: x.replace(prefix, ''))

    # Skip everything after the first dot ('.').
    df['url'] = df['url'].apply(lambda x: x.replace(prefix, ''))

    return df




def load_and_preprocess_reviews(path='train_reviews.json'):
    """
    :param path: Path to the json file.
    :return df: A preprocessed pandas dataframe.
    """

    df = pd.read_json(path)

    # Format reviews column to contain only a single int, the stars/rating.
    df['stars'] = df['stars'].apply(lambda x: x[24])

    # Find which company each review concerns, based on the url that it got submitted to.
    df = match_url_to_company(df)


    return df


if __name__ == '__main__':

    reviews = load_and_preprocess_reviews()

    print(reviews['url'].unique())

    pass