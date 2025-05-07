# MovieLens Dataset

This directory is intended for storing the MovieLens dataset files. The application requires at least the following files:

- `ratings.csv`: Contains user ratings for movies
- `movies.csv`: Contains movie information (titles and genres)

## Downloading the Dataset

You can download the MovieLens dataset from the GroupLens website:

1. Visit [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)
2. Choose one of the dataset versions:

   - ml-latest-small (100,000 ratings, 9,000 movies) - Recommended for getting started
   - ml-25m (25 million ratings, 62,000 movies) - For more comprehensive analysis
   - ml-latest (Latest full dataset) - For production applications

3. Download and extract the ZIP file
4. Copy at least the `ratings.csv` and `movies.csv` files to this directory

## Dataset Format

### ratings.csv

Contains the following columns:

- `userId`: Unique identifier for each user
- `movieId`: Unique identifier for each movie
- `rating`: Rating given by the user (scale of 0.5-5 in 0.5 step increments)
- `timestamp`: Timestamp of when the rating was given

### movies.csv

Contains the following columns:

- `movieId`: Unique identifier for each movie
- `title`: Movie title with release year in parentheses
- `genres`: Pipe-separated list of genres (e.g., "Action|Adventure|Sci-Fi")

## Notes

- The application supports sampling for large datasets. If you're using one of the larger datasets (like ml-25m), you can adjust the sample size in the application.
- The minimum number of ratings per user and per movie can be configured in the application to filter out users and movies with too few ratings.
