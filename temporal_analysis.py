"""
Alexandros Metsai
alexmetsai@gmail.com

Various utilities used for our analysis.
"""

import matplotlib.pyplot as plt

from utils.utils import load_and_preprocess_reviews


def average_rating_per_year(df):
    """
    Group the DateFrame by year and plot the average rating for each year.
    """

    df['year'] = df['date'].apply(lambda x: x.year)

    grouped_df = df.groupby('year')
    mean_df = grouped_df['stars'].mean()
    ax = mean_df.plot.barh(x='year', y='stars')
    ax.set_xlabel("Average Rating")
    plt.show()

    return df


if __name__ == '__main__':

    reviews = load_and_preprocess_reviews()
    print(f"Earliest date:{reviews.date.min()}\nLatest date{reviews.date.max()}")

    avg = average_rating_per_year(reviews)

