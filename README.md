# Netflix-Style Movie Recommendation System

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Algorithm Explanation](#Algorithm-explanation)
  - [Popularity Based Algorithm](#1-popularity-based-recommendations)
  - [Content Based Algorithm](#2-content-based-filtering)
  - [Collaborative Filtering Algorithm](#3-collaborative-filtering)
  - [Item Based Algorithm](#4-item-based-collaborative-filtering)
  - [Hybrid Algorithm](#5-hybrid-recommendations)
- [Improvements](#improvements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This project is a Streamlit web application that provides a Netflix-style movie recommendation interface. It allows users to upload MovieLens datasets and explore different recommendation algorithms..

## Features

- **Multiple recommendation algorithms**:

  - **Popularity-based recommendations**: Ranks movies by the number of ratings and average ratings.
  - **Content-based filtering**: Recommends movies similar to a selected one based on genres using cosine similarity.
  - **Collaborative filtering**: Uses matrix factorization (SVD) to predict user ratings and recommend movies based on similar users' preferences
  - **Item-based collaborative filtering**: Finds movies similar to a selected one based on user rating patterns using k-Nearest Neighbors.
  - **Hybrid recommendations**: ombines collaborative filtering with content similarity for more personalized results.

- **Interactive Streamlit interface** with:

  - Dataset exploration and visualization
  - Personalized recommendations for users
  - Movie similarity analysis
  - Genre distribution visualization
  - Comparison of recommendation algorithms

- **Robust implementation** with:
  - Comprehensive error handling
  - Session state management for caching data and models
  - Performance optimizations for large datasets

## Installation

### Prerequisites

- Python 3.7+
- pip

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/netflix-recommendation-system.git
cd netflix-recommendation-system
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Download the MovieLens dataset:
   - Visit [grouplens.org/datasets/movielens](https://grouplens.org/datasets/movielens/)
   - Download the dataset of your choice (recommend starting with "ml-latest-small")
   - Extract the ZIP file and place the CSV files in the `data` directory

## Usage

1. Start the Streamlit app:

```bash
streamlit run app.py
```

2. Open your web browser and go to [http://localhost:8501](http://localhost:8501)

3. In the app:
   - Enter the paths to your MovieLens CSV files
   - Click "Load Data" to process the dataset
   - Navigate through the tabs to explore different recommendation methods

## Project Structure

```
movie-recommendation-system/
│
├── app.py                  # Main Streamlit application
├── recommendation/         # Recommendation algorithms modules
│   ├── __init__.py
│   ├── content_based.py    # Content-based filtering functions
│   ├── collaborative.py    # Collaborative filtering functions
│   ├── item_based.py       # Item-based collaborative filtering
│   ├── popularity.py       # Popularity-based recommendations
│   └── hybrid.py           # Hybrid recommendation functions
│
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── data_loader.py      # Functions for loading and preprocessing data
│   └── visualization.py    # Functions for creating visualizations
│
├── data/                   # Directory for MovieLens dataset files
│   ├── README.md           # Instructions for downloading the dataset
│   └── .gitkeep
│
├── images/                 # Images for README and documentation
│
├── requirements.txt        # Required Python packages
└── README.md               # Project documentation
```

## Algorithms Explained

### 1. Popularity-Based Recommendations

The simplest approach that recommends movies with the highest number of ratings and/or highest average ratings. Great for new users with no rating history.

### 2. Content-Based Filtering

Recommends movies similar to ones a user has liked based on movie features (genres in our implementation). Uses cosine similarity to find movies with similar genre profiles.

### 3. Collaborative Filtering

Identifies patterns in user preferences and predicts ratings based on how similar users have rated items. We implement matrix factorization using Singular Value Decomposition (SVD).

### 4. Item-Based Collaborative Filtering

Finds movies that are similar based on user rating patterns rather than content features. Uses k-Nearest Neighbors to identify movies that users tend to rate similarly.

### 5. Hybrid Recommendations

Combines collaborative filtering predictions with content-based similarity to provide more robust recommendations, overcoming the limitations of individual approaches.

## Improvements

- Add more features (actors, directors, plot keywords) for content-based filtering
- Implement deep learning recommendation models
- Add user registration and personalized profiles
- Incorporate movie posters and trailers
- Add evaluation metrics to compare algorithm performance
- Optimize for larger datasets with caching and more efficient computation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MovieLens](https://grouplens.org/datasets/movielens/) for providing the dataset
- [Streamlit](https://streamlit.io/) for the amazing web application framework
- [scikit-learn](https://scikit-learn.org/) for machine learning algorithms
- [Netflix](https://research.netflix.com/) for inspiring this recommendation system
- [Sundeep Yalamanchili](https://github.com/Yalamanchili7) for providing guide to create the project
