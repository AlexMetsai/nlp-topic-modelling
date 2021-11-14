"""
Alexandros Metsai
alexmetsai@gmail.com

Analyze ratings over time.
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


def average_rating_per_month(df):
    """
    Plots the average rating of each month, for all the years.
    """

    df['month'] = df['date'].apply(lambda x: x.month)

    grouped_df = df.groupby('month')
    mean_df = grouped_df['stars'].mean()
    ax = mean_df.plot.barh(x='month', y='stars')
    ax.set_xlabel("Average Rating")
    plt.show()

    return df


def average_rating_monthly(df):
    """
    Plots the average rating as time passes, in a monthly rate (2016-12, 2017-01, etc.)
    """

    df['month'] = df['date'].apply(lambda x: f"{x.year}  {x.month}")

    grouped_df = df.groupby('month')
    mean_df = grouped_df['stars'].mean()
    ax = mean_df.plot.barh(x='month', y='stars')
    ax.set_xlabel("Average Rating")
    plt.show()

    return df


if __name__ == '__main__':

    reviews = load_and_preprocess_reviews()
    print(f"Earliest date:{reviews.date.min()}\nLatest date{reviews.date.max()}")

    avg_year = average_rating_per_year(reviews)
    avg_month = average_rating_per_month(reviews)

