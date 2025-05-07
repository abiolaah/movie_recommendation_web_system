from recommendation.collaborative import get_cf_recommendations


def get_hybrid_recommendations(user_id, movies, predicted_df, user_item_matrix, movie_id=None, n=10):
    """Get hybrid recommendations combining collaborative filtering and content-based """
    
    # Get collaborative filtering recommendations
    cf_recs = get_cf_recommendations(user_id, predicted_df, user_item_matrix, movies, n=n*2)
    
    # If no movie_id provided, return CF recommendations
    if movie_id is None:
        return cf_recs.head(n)
    
    # Blend the scores using genre similarity
    input_movie = movies[movies['movieId']== movie_id]
    
    if not input_movie.empty:
        input_genres = set(input_movie.iloc[0]['genres'].split('|'))
        
        # Calculate genre similarity for each recommendation
        cf_recs['genre_match'] = cf_recs['genres'].apply(
            lambda x: len(set(x.split('|')) & input_genres) / len(input_genres)
            if len(input_genres) > 0 else 0
        )
        
        # Adjust predicted rating based on genre similarity
        cf_recs['hybrid_score'] =(
            0.7 * cf_recs['predicted_rating'] +
            0.3 * cf_recs['genre_match']
        )
        
        # Sort by hybrid score
        hybrid_recs = cf_recs.sort_values(by='hybrid_score', ascending=False).head(n)
        return hybrid_recs
    else:
        return cf_recs.head(n)  # Return CF recommendations if input movie not found