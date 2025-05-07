# utils/visualization.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_rating_distribution(ratings):
    """Plot the distribution of ratings."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ratings['rating'].value_counts().sort_index().plot(kind='bar', ax=ax)
    ax.set_xlabel('Rating')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Movie Ratings')
    return fig

def plot_genre_distribution(movies):
    """Plot the distribution of genres."""
    # Split genres and count
    all_genres = movies['genres'].str.split('|').explode()
    genre_counts = all_genres.value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    genre_counts.plot(kind='bar', ax=ax)
    ax.set_xlabel('Genre')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Movie Genres')
    return fig