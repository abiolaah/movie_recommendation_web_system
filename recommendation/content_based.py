import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def create_content_based_recommender(movies):
    """Create content-based recommendation system based on genres"""
    # Create a matrix of genres
    genres_list = []
    for genres in movies['genres']:
        genres_list.extend(genres.split('|'))
        
    unique_genres = sorted(list(set(genres_list)))
    
    # One-hot encode the genres
    genre_matrix = np.zeros((len(movies), len(unique_genres)))
    
    for i, genres in enumerate(movies['genres']):
        for genre in genres.split('|'):
            if genre in unique_genres:
                idx = unique_genres.index(genre)
                genre_matrix[i, idx] = 1
                
    # Compute cosine similarity
    cosine_sim = cosine_similarity(genre_matrix)
    
    # Create a movie title to index mapping
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    
    return cosine_sim, indices

def get_content_recommendations(title, cosine_sim, indices, movies, n=10):
    """Get movie recommendations based on content similarity"""
    # Get the index of the movie
    idx = indices[title]
    
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort by similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top N movie indices
    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the recommended movies with similarity scores
    result = movies.iloc[movie_indices].copy()
    result['similarity'] = [score for _, score in sim_scores]
    
    return result