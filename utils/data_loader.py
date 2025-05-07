import pandas as pd

# function to load data
def load_data(ratings_path, movies_path, sample_size=None):
    """Lod ratings and movies data from MovieLens datasets"""
    try:
        # Load ratings
        ratings = pd.read_csv(ratings_path)
        
        # Sample data if requested
        if sample_size and len(ratings) > sample_size:
            ratings = ratings.sample(sample_size, random_state=42)
            
        # Load movies
        movies = pd.read_csv(movies_path)
        
        return ratings, movies
    except Exception as e:
        print(f"Error in load_data: {e}")
        return None, None
    

# function to preprocess data
def preprocess_data(ratings, movies, min_user_rating=10, min_movie_ratings=10):
    """Preprocess ratings and movies data"""
    # Merge ratings and movies data
    data = pd.merge(ratings, movies, on='movieId')
    
    # Filter out users with few ratings
    user_counts = data.groupby('userId')['rating'].count()
    active_users = user_counts[user_counts >= min_user_rating].index
    data = data[data['userId'].isin(active_users)]
    
    # Filter out movies with few ratings
    movie_counts = data.groupby('movieId')['rating'].count()
    active_movies = movie_counts[movie_counts >=min_movie_ratings].index
    data = data[data['movieId'].isin(active_movies)]
    
    # Create user-item matrix
    user_item_matrix = data.pivot(index='userId', columns='movieId', values='rating')
    
    # Fill missing values with 0
    user_item_matrix_filled = user_item_matrix.fillna(0)
    
    return data, user_item_matrix, user_item_matrix_filled