"""
Alexandros Metsai
alexmetsai@gmail.com

Initial analysis of the Trustpilot dataset.
"""


import pandas as pd
import matplotlib.pyplot as plt

from utils.utils import load_and_preprocess_reviews


if __name__ == '__main__':

    reviews = load_and_preprocess_reviews()

    # Plot the number of reviews for each company.
    reviews['company'].value_counts(sort=True).plot.barh()
    plt.show()

    # Plot the average stars for each company.
    grouped_df = reviews.groupby("company")
    mean_df = grouped_df['stars'].mean()
    mean_df = mean_df.reset_index()
    mean_df.plot.barh(x='company', y='stars')
    plt.show()

    # For each company plot together Average Stars and Number of Reviews.
    # We plot the two axes with different scaling.
    review_mean_and_count = reviews['company'].value_counts(sort=True).rename_axis('company').reset_index(name='counts')
    review_mean_and_count = pd.merge(review_mean_and_count, mean_df)
    review_mean_and_count.set_index('company').plot(kind='bar', secondary_y='stars', rot=90)
    plt.show()

