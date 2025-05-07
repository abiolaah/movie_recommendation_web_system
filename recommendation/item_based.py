import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

def create_item_based_recommender(user_item_matrix):
    """Create item-based collaborative filtering model"""
    # Fill NaNs with 0 (assume unrated = 0)
    user_item_matrix_filled = user_item_matrix.fillna(0)
    # Convert the user-item matrix to a sparse matrix
    sparse_user_item = csr_matrix(user_item_matrix_filled)
    # Train the model
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
    model_knn.fit(sparse_user_item.T) # Transpose for item-based
    return model_knn

def get_similar_movies(movie_id, user_item_matrix, model_knn, movies, n=10):
    """Get similar movies for a given movie"""
    # Find the index of the movie in the user_item_matrix columns
    if movie_id not in user_item_matrix.columns:
        return pd.DataFrame() # Movie not found
    
    movie_idx = list(user_item_matrix.columns).index(movie_id)
    
    # Get the feature vector for the movie
    movie_vector = user_item_matrix.iloc[:, movie_idx].values.reshape(1,-1)
    
    # Find n+1 neighbors (including itself)
    distances, indices = model_knn.kneighbors(movie_vector, n_neighbors=n+1)
    
    # Get movie indices (skip the first, which is the movie itself)
    similar_indices = indices.flatten()[1:]
    similarity_scores = 1 - distances.flatten()[1:]  # cosine similarity
    similar_movie_ids = [user_item_matrix.columns[i] for i in similar_indices]
    # Prepare result DataFrame
    recs = movies[movies['movieId'].isin(similar_movie_ids)].copy()
    recs = recs.set_index('movieId').loc[similar_movie_ids].reset_index()
    recs['similarity'] = similarity_scores
    recs = recs[['title', 'genres', 'similarity']]
    return recs