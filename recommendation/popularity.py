import pandas as pd

def get_popular_movies(ratings, movies, n=10):
    """Get the most popular movies based on number of ratings"""
    # Count ratings per movie
    movie_counts = ratings.groupby('movieId').size().reset_index(name='count')
    
    # Calculate average rating
    movie_ratings = ratings.groupby('movieId')['rating'].mean().reset_index(name='avg_rating')
    
    # Merge counts and ratings
    movie_stats = pd.merge(movie_counts, movie_ratings, on='movieId')
    
    # Merge with movie details
    popular_movies = pd.merge(movie_stats, movies, on='movieId')
    
    # Sort by popularity
    popular_movies = popular_movies.sort_values('count', ascending=False)
    
    return popular_movies.head(n)