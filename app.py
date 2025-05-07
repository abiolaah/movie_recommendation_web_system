# app.py - Main Streamlit application

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import utility functions
from utils.data_loader import load_data, preprocess_data
from utils.visualization import plot_rating_distribution, plot_genre_distribution

# Import recommendation algorithms
from recommendation.popularity import get_popular_movies
from recommendation.content_based import create_content_based_recommender, get_content_recommendations
from recommendation.collaborative import create_collaborative_filtering, get_cf_recommendations
from recommendation.item_based import create_item_based_recommender, get_similar_movies
from recommendation.hybrid import get_hybrid_recommendations

# Set page config
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

def main():
    st.title("🎬 Netflix-Style Movie Recommendation System")
    
    st.write("""
    This app provides personalized movie recommendations using algorithms similar to those used by Netflix.
    Upload your MovieLens dataset and explore different recommendation approaches.
    """)
    
    # Sidebar for data loading and configuration
    st.sidebar.header("Dataset Configuration")
    
    # Input paths for dataset files
    ratings_path = st.sidebar.text_input(
        "Ratings CSV Path", 
        value="data/ratings.csv"
    )
    
    movies_path = st.sidebar.text_input(
        "Movies CSV Path", 
        value="data/movies.csv"
    )
    
    # Sample size option
    sample_size = st.sidebar.slider(
        "Sample Size (set to 0 for full dataset)", 
        min_value=0, 
        max_value=1000000, 
        value=100000, 
        step=10000
    )
    
    sample_size = sample_size if sample_size > 0 else None
    
    # Minimum ratings filters
    min_user_ratings = st.sidebar.slider("Min Ratings per User", 5, 100, 20)
    min_movie_ratings = st.sidebar.slider("Min Ratings per Movie", 5, 100, 20)
    
    # Load data button
    if st.sidebar.button("Load Data"):
        st.session_state.clear()  # Clear the session state to avoid cached errors
        
        with st.spinner("Loading and preprocessing data... This may take a while."):
            try:
                # Load the data
                ratings, movies = load_data(ratings_path, movies_path, sample_size)
                
                if ratings is not None and movies is not None:
                    # Store in session state
                    st.session_state.ratings = ratings
                    st.session_state.movies = movies
                    
                    # Preprocess data
                    data, user_item_matrix, user_item_matrix_filled = preprocess_data(
                        ratings, movies, min_user_ratings, min_movie_ratings
                    )
                    
                    if data is not None:
                        st.session_state.data = data
                        st.session_state.user_item_matrix = user_item_matrix
                        st.session_state.user_item_matrix_filled = user_item_matrix_filled
                        
                        # Create recommendation models
                        with st.spinner("Building recommendation models..."):
                            # Content-based model
                            cosine_sim, indices = create_content_based_recommender(movies)
                            st.session_state.cosine_sim = cosine_sim
                            st.session_state.indices = indices
                            
                            # Collaborative filtering model
                            predicted_df = create_collaborative_filtering(user_item_matrix_filled)
                            st.session_state.predicted_df = predicted_df
                            
                            # Item-based collaborative filtering
                            model_knn = create_item_based_recommender(user_item_matrix)
                            st.session_state.model_knn = model_knn
                        
                        st.session_state.data_loaded = True
                        st.sidebar.success("Data loaded and models built successfully!")
                    else:
                        st.sidebar.error("Error preprocessing data.")
                        st.session_state.data_loaded = False
                else:
                    st.sidebar.error("Failed to load data.")
                    st.session_state.data_loaded = False
            
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
                st.session_state.data_loaded = False
    
    # Create tabs for different features
    tabs = st.tabs([
        "Dataset Overview", 
        "Popular Movies", 
        "Content-Based", 
        "Collaborative Filtering",
        "Item-Based",
        "Hybrid Recommendations"
    ])
    
    # Dataset Overview Tab
    with tabs[0]:
        st.header("Dataset Overview")
        
        # Check if required session state variables exist
        if not all(key in st.session_state for key in ["ratings", "movies"]):
            st.warning("Please load the dataset first by clicking the 'Load Data' button in the sidebar.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Ratings Dataset")
                st.write(f"Shape: {st.session_state.ratings.shape}")
                st.dataframe(st.session_state.ratings.head())
            
            with col2:
                st.subheader("Movies Dataset")
                st.write(f"Shape: {st.session_state.movies.shape}")
                st.dataframe(st.session_state.movies.head())
            
            # Plot rating distribution
            st.subheader("Rating Distribution")
            try:
                fig_ratings = plot_rating_distribution(st.session_state.ratings)
                st.pyplot(fig_ratings)
            except Exception as e:
                st.error(f"Error plotting rating distribution: {str(e)}")
                
            # Plot genre distribution    
            st.subheader("Most Common Genres")
            try:
                fig_genres = plot_genre_distribution(st.session_state.movies)
                st.pyplot(fig_genres)
            except Exception as e:
                st.error(f"Error plotting genre distribution: {str(e)}")
    
    # Popular Movies Tab
    with tabs[1]:
        st.header("Most Popular Movies")
        
        # Check if required session state variables exist
        if not all(key in st.session_state for key in ["ratings", "movies"]):
            st.warning("Please load the dataset first by clicking the 'Load Data' button in the sidebar.")
        else:
            try:
                n_popular = st.slider("Number of popular movies to show", 5, 50, 10)
                
                popular_movies = get_popular_movies(
                    st.session_state.ratings, 
                    st.session_state.movies, 
                    n=n_popular
                )
                
                if not popular_movies.empty:
                    st.dataframe(
                        popular_movies[['title', 'genres', 'count', 'avg_rating']],
                        column_config={
                            "title": "Movie Title",
                            "genres": "Genres",
                            "count": st.column_config.NumberColumn(
                                "Number of Ratings",
                                format="%d"
                            ),
                            "avg_rating": st.column_config.NumberColumn(
                                "Average Rating",
                                format="%.2f ⭐"
                            )
                        },
                        hide_index=True
                    )
                    
                    # Plot top 10 popular movies
                    st.subheader("Top Movies by Popularity")
                    fig, ax = plt.subplots(figsize=(12, 8))
                    top_10 = popular_movies.head(10)
                    ax.barh(top_10['title'], top_10['count'], color='skyblue')
                    ax.set_xlabel('Number of Ratings')
                    ax.set_title('Top 10 Most Popular Movies')
                    ax.invert_yaxis()
                    st.pyplot(fig)
                else:
                    st.error("Failed to retrieve popular movies.")
            except Exception as e:
                st.error(f"Error in popular movies tab: {str(e)}")
    
    # Content-Based Tab
    with tabs[2]:
        st.header("Content-Based Recommendations")
        st.write("""
        Content-based recommendations suggest movies similar to ones you already like,
        based on features like genre, actors, or directors. Here we're using movie genres.
        """)
        
        # Check if required session state variables exist
        if not all(key in st.session_state for key in ["movies", "cosine_sim", "indices"]):
            st.warning("Please load the dataset first by clicking the 'Load Data' button in the sidebar.")
        else:
            try:
                # Movie selection
                movie_titles = st.session_state.movies['title'].tolist()
                selected_movie = st.selectbox("Select a movie you like:", movie_titles)
                
                if st.button("Get Similar Movies (Content-Based)"):
                    with st.spinner("Finding similar movies..."):
                        similar_movies = get_content_recommendations(
                            selected_movie,
                            st.session_state.cosine_sim,
                            st.session_state.indices,
                            st.session_state.movies
                        )
                        
                        if not similar_movies.empty:
                            st.dataframe(
                                similar_movies[['title', 'genres', 'similarity']],
                                column_config={
                                    "title": "Movie Title",
                                    "genres": "Genres",
                                    "similarity": st.column_config.NumberColumn(
                                        "Similarity Score",
                                        format="%.3f"
                                    )
                                },
                                hide_index=True
                            )
                            
                            # Highlight common genres
                            st.subheader("Genre Analysis")
                            selected_genres = set(st.session_state.movies[
                                st.session_state.movies['title'] == selected_movie
                            ]['genres'].iloc[0].split('|'))
                            
                            st.write(f"Genres of '{selected_movie}': {', '.join(sorted(selected_genres))}")
                            
                            # Count genre matches in recommendations
                            genre_matches = []
                            for _, row in similar_movies.iterrows():
                                rec_genres = set(row['genres'].split('|'))
                                common = rec_genres.intersection(selected_genres)
                                genre_matches.extend(list(common))
                            
                            match_counts = pd.Series(genre_matches).value_counts()
                            
                            # Plot genre matches
                            fig, ax = plt.subplots(figsize=(10, 6))
                            match_counts.plot(kind='bar', ax=ax)
                            ax.set_title('Common Genres in Recommendations')
                            ax.set_xlabel('Genre')
                            ax.set_ylabel('Count')
                            plt.xticks(rotation=45)
                            st.pyplot(fig)
                        else:
                            st.error("Could not find similar movies for this title.")
            except Exception as e:
                st.error(f"Error in content-based recommendations: {str(e)}")
    
    # Collaborative Filtering Tab
    with tabs[3]:
        st.header("Collaborative Filtering Recommendations")
        st.write("""
        Collaborative filtering suggests movies based on similar user preferences.
        It finds patterns in user behavior without needing to know about movie content.
        """)
        
        # Check if required session state variables exist
        if not all(key in st.session_state for key in ["predicted_df", "user_item_matrix", "movies", "data"]):
            st.warning("Please load the dataset first by clicking the 'Load Data' button in the sidebar.")
        else:
            try:
                # User selection
                user_ids = sorted(st.session_state.user_item_matrix.index.tolist())
                selected_user = st.selectbox("Select a user ID:", user_ids)
                
                n_recommendations = st.slider(
                    "Number of recommendations", 
                    min_value=5, 
                    max_value=20, 
                    value=10
                )
                
                if st.button("Get Recommendations (Collaborative Filtering)"):
                    with st.spinner("Generating personalized recommendations..."):
                        cf_recs = get_cf_recommendations(
                            selected_user,
                            st.session_state.predicted_df,
                            st.session_state.user_item_matrix,
                            st.session_state.movies,
                            n=n_recommendations
                        )
                        
                        if not cf_recs.empty:
                            st.dataframe(
                                cf_recs[['title', 'genres', 'predicted_rating']],
                                column_config={
                                    "title": "Movie Title",
                                    "genres": "Genres",
                                    "predicted_rating": st.column_config.NumberColumn(
                                        "Predicted Rating",
                                        format="%.2f ⭐"
                                    )
                                },
                                hide_index=True
                            )
                            
                            # Show user's actual ratings for comparison
                            st.subheader("Movies This User Has Already Rated")
                            user_data = st.session_state.data[
                                st.session_state.data['userId'] == selected_user
                            ].sort_values('rating', ascending=False)
                            
                            st.dataframe(
                                user_data[['title', 'genres', 'rating']],
                                column_config={
                                    "title": "Movie Title",
                                    "genres": "Genres",
                                    "rating": st.column_config.NumberColumn(
                                        "User Rating",
                                        format="%.1f ⭐"
                                    )
                                },
                                hide_index=True
                            )
                            
                            # Genre analysis of recommendations
                            st.subheader("Genre Distribution in Recommendations")
                            
                            # Count genres in recommendations
                            rec_genres = []
                            for g in cf_recs['genres']:
                                rec_genres.extend(g.split('|'))
                            
                            rec_genre_counts = pd.Series(rec_genres).value_counts()
                            
                            # Count genres in user's highly rated movies
                            user_genres = []
                            for g in user_data[user_data['rating'] >= 4]['genres']:
                                user_genres.extend(g.split('|'))
                            
                            user_genre_counts = pd.Series(user_genres).value_counts()
                            
                            # Plot comparison
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                            
                            # Plot recommended genres
                            rec_genre_counts.head(10).plot(kind='bar', ax=ax1)
                            ax1.set_title('Genres in Recommendations')
                            ax1.set_xlabel('Genre')
                            ax1.set_ylabel('Count')
                            plt.sca(ax1)
                            plt.xticks(rotation=45)
                            
                            # Plot user's favorite genres
                            user_genre_counts.head(10).plot(kind='bar', ax=ax2)
                            ax2.set_title('User\'s Favorite Genres')
                            ax2.set_xlabel('Genre')
                            ax2.set_ylabel('Count')
                            plt.sca(ax2)
                            plt.xticks(rotation=45)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.error("Could not generate recommendations for this user.")
            except Exception as e:
                st.error(f"Error in collaborative filtering recommendations: {str(e)}")
    
    # Item-Based Tab
    with tabs[4]:
        st.header("Item-Based Collaborative Filtering")
        st.write("""
        Item-based collaborative filtering finds movies that are similar based on user rating patterns.
        Select a movie to find others that users tend to rate similarly.
        """)
        
        # Check if required session state variables exist
        if not all(key in st.session_state for key in ["user_item_matrix", "model_knn", "movies"]):
            st.warning("Please load the dataset first by clicking the 'Load Data' button in the sidebar.")
        else:
            try:
                # Get movie IDs that exist in the matrix
                available_movie_ids = set(st.session_state.user_item_matrix.columns.tolist())
                
                # Filter to only show movies that are in the user-item matrix
                available_movies = st.session_state.movies[
                    st.session_state.movies['movieId'].isin(available_movie_ids)
                ]
                
                if available_movies.empty:
                    st.error("No movies with sufficient ratings were found in the preprocessed data.")
                else:
                    # Movie selection - sorted by title for easier selection
                    movie_titles = sorted(available_movies['title'].tolist())
                    selected_title = st.selectbox("Select a movie:", movie_titles)
                    
                    # Get movieId for the selected title
                    movie_row = available_movies[available_movies['title'] == selected_title]
                    
                    if movie_row.empty:
                        st.error("Selected movie not found in available movies.")
                    else:
                        selected_movie_id = movie_row['movieId'].iloc[0]
                        
                        # Check if the movie is in the matrix again (just to be safe)
                        if selected_movie_id not in available_movie_ids:
                            st.error(f"Movie '{selected_title}' (ID: {selected_movie_id}) is not in the user-item matrix.")
                        else:
                            if st.button("Find Similar Movies (Item-Based)"):
                                with st.spinner("Finding similar movies..."):
                                    similar_movies = get_similar_movies(
                                        selected_movie_id,
                                        st.session_state.user_item_matrix,
                                        st.session_state.model_knn,
                                        st.session_state.movies
                                    )
                                    
                                    if not similar_movies.empty:
                                        st.dataframe(
                                            similar_movies[['title', 'genres', 'similarity']],
                                            column_config={
                                                "title": "Movie Title",
                                                "genres": "Genres",
                                                "similarity": st.column_config.NumberColumn(
                                                    "Similarity Score",
                                                    format="%.3f"
                                                )
                                            },
                                            hide_index=True
                                        )
                                        
                                        # Show genre overlap
                                        st.subheader("Genre Comparison")
                                        
                                        # Get selected movie genres
                                        selected_movie_genres = set(movie_row['genres'].iloc[0].split('|'))
                                        
                                        # Calculate genre overlap for each similar movie
                                        genre_overlaps = []
                                        titles = []
                                        
                                        for _, row in similar_movies.iterrows():
                                            rec_genres = set(row['genres'].split('|'))
                                            overlap = len(rec_genres.intersection(selected_movie_genres)) / max(1, len(selected_movie_genres))
                                            genre_overlaps.append(overlap * 100)  # Convert to percentage
                                            titles.append(row['title'])
                                        
                                        # Plot genre overlap
                                        if titles:  # Make sure we have titles before plotting
                                            fig, ax = plt.subplots(figsize=(12, 6))
                                            ax.barh(titles, genre_overlaps, color='skyblue')
                                            ax.set_xlabel('Genre Overlap (%)')
                                            ax.set_title(f'Genre Overlap with "{selected_title}"')
                                            ax.invert_yaxis()  # Show first recommendation at the top
                                            st.pyplot(fig)
                                    else:
                                        st.warning("No similar movies found for this title.")
            except Exception as e:
                st.error(f"Error in item-based recommendations tab: {str(e)}")
    
    # Hybrid Recommendations Tab
    with tabs[5]:
        st.header("Hybrid Recommendations")
        st.write("""
        Hybrid recommendations combine multiple approaches for better results.
        Select a user and optionally a movie they like to get personalized recommendations
        that blend collaborative filtering with content similarity.
        """)
        
        # Check if required session state variables exist
        if not all(key in st.session_state for key in ["user_item_matrix", "predicted_df", "movies", "data"]):
            st.warning("Please load the dataset first by clicking the 'Load Data' button in the sidebar.")
        else:
            try:
                # User selection
                user_ids = sorted(st.session_state.user_item_matrix.index.tolist())
                selected_user = st.selectbox("Select a user ID (for hybrid recommendations):", user_ids)
                
                # Optional movie selection
                st.write("Optionally select a movie to influence recommendations:")
                
                # Get movies this user has rated highly
                user_ratings = st.session_state.data[
                    st.session_state.data['userId'] == selected_user
                ].sort_values('rating', ascending=False)
                
                if not user_ratings.empty:
                    st.write("Top rated movies by this user:")
                    st.dataframe(
                        user_ratings.head(5)[['title', 'rating']],
                        column_config={
                            "title": "Movie Title",
                            "rating": st.column_config.NumberColumn(
                                "Rating",
                                format="%.1f ⭐"
                            )
                        },
                        hide_index=True
                    )
                
                # All available movies that exist in the matrix
                available_movie_ids = st.session_state.user_item_matrix.columns.tolist()
                available_movies = st.session_state.movies[
                    st.session_state.movies['movieId'].isin(available_movie_ids)
                ]
                
                # Movie selection
                movie_titles = ["None"] + sorted(available_movies['title'].tolist())
                selected_title = st.selectbox("Select a movie (optional):", movie_titles)
                
                # Get movieId for the selected title
                selected_movie_id = None
                if selected_title != "None":
                    movie_row = available_movies[available_movies['title'] == selected_title]
                    if not movie_row.empty:
                        selected_movie_id = movie_row['movieId'].iloc[0]
                
                if st.button("Get Hybrid Recommendations"):
                    with st.spinner("Generating personalized hybrid recommendations..."):
                        hybrid_recs = get_hybrid_recommendations(
                            selected_user,
                            st.session_state.movies,
                            st.session_state.predicted_df,
                            st.session_state.user_item_matrix,
                            movie_id=selected_movie_id
                        )
                        
                        if not hybrid_recs.empty:
                            # Display columns based on whether a movie was selected
                            if selected_movie_id is not None:
                                display_cols = ['title', 'genres', 'predicted_rating', 'genre_match', 'hybrid_score']
                                column_config = {
                                    "title": "Movie Title",
                                    "genres": "Genres",
                                    "predicted_rating": st.column_config.NumberColumn(
                                        "CF Score", 
                                        format="%.2f ⭐"
                                    ),
                                    "genre_match": st.column_config.NumberColumn(
                                        "Genre Match",
                                        format="%.2f"
                                    ),
                                    "hybrid_score": st.column_config.NumberColumn(
                                        "Hybrid Score",
                                        format="%.2f"
                                    )
                                }
                            else:
                                display_cols = ['title', 'genres', 'predicted_rating']
                                column_config = {
                                    "title": "Movie Title",
                                    "genres": "Genres",
                                    "predicted_rating": st.column_config.NumberColumn(
                                        "Predicted Rating",
                                        format="%.2f ⭐"
                                    )
                                }
                            
                            st.dataframe(
                                hybrid_recs[display_cols],
                                column_config=column_config,
                                hide_index=True
                            )
                            
                            # Create a visualization of the recommendations
                            if selected_movie_id is not None and 'genre_match' in hybrid_recs.columns:
                                st.subheader("Recommendation Factors")
                                
                                # Plot the factors that influenced recommendations
                                fig, ax = plt.subplots(figsize=(12, 8))
                                
                                x = range(len(hybrid_recs))
                                width = 0.35
                                
                                # Normalize scores for better visualization
                                cf_scores = hybrid_recs['predicted_rating'] / 5  # Assuming max rating is 5
                                genre_scores = hybrid_recs['genre_match']
                                
                                # Plot stacked bars
                                ax.bar(x, cf_scores, width, label='Collaborative Filtering')
                                ax.bar(x, genre_scores, width, bottom=cf_scores, label='Genre Similarity')
                                
                                ax.set_ylabel('Score Contribution')
                                ax.set_title('Factors Influencing Hybrid Recommendations')
                                ax.set_xticks(x)
                                ax.set_xticklabels(hybrid_recs['title'], rotation=45, ha='right')
                                ax.legend()
                                
                                plt.tight_layout()
            except Exception as e:
                st.error(f"An error occurred: {e}")
                
if __name__ == "__main__":
    main()