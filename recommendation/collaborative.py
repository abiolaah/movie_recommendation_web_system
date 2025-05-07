import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


def create_collaborative_filtering(user_item_matrix_filled, n_components=50):
    """Create collaborative filtering recommender using SVD"""
    
    # Create SVD model
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    
    # Fit the model
    latent_matrix = svd.fit_transform(user_item_matrix_filled)
    
    # Get item latent factors
    item_latent_factors = svd.components_
    
    # Compute predicted ratings
    predicted_ratings = np.dot(latent_matrix, item_latent_factors)
    
    # Create DataFrame with predicted ratings
    predicted_df = pd.DataFrame(
        predicted_ratings,
        index=user_item_matrix_filled.index,
        columns=user_item_matrix_filled.columns
    )
    
    return predicted_df

def get_cf_recommendations(user_id, predicted_df, user_item_matrix, movies, n):
    """Get top N movie recommendations for a user using collaborative filtering"""
    # Get predicted ratings for the user
    user_pred = predicted_df.loc[user_id]
    # Get movies the user has already rated
    rated_movies = user_item_matrix.loc[user_id]
    already_rated = rated_movies[rated_movies.notna()].index
    # Exclude already rated movies
    user_pred = user_pred.drop(already_rated, errors='ignore')
    # Get top N recommendations
    top_n = user_pred.sort_values(ascending=False).head(n)
    # Prepare result DataFrame
    recs = movies[movies['movieId'].isin(top_n.index)].copy()
    recs = recs.set_index('movieId').loc[top_n.index].reset_index()
    recs['predicted_rating'] = top_n.values
    recs = recs[['title', 'genres', 'predicted_rating']]
    return recs