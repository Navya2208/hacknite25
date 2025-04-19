import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def clean_text(text):
    """
    Clean text by removing special characters, extra spaces, etc.
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text
    """
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Remove special characters
    text = text.lower()
    text = ' '.join(text.split())
    
    return text

def extract_year_from_date(date_string):
    """
    Extract year from date string.
    
    Args:
        date_string (str): Date string
        
    Returns:
        int or None: Extracted year or None if extraction fails
    """
    try:
        return pd.to_datetime(date_string).year
    except:
        return None

def split_genres(genres_string, delimiter=','):
    """
    Split genres string into list of genres.
    
    Args:
        genres_string (str): String of genres
        delimiter (str): Delimiter character
        
    Returns:
        list: List of genres
    """
    if pd.isna(genres_string) or genres_string is None:
        return []
    
    return [g.strip() for g in genres_string.split(delimiter) if g.strip()]

def extract_duration_info(duration_string):
    """
    Extract duration value and type from duration string.
    
    Args:
        duration_string (str): Duration string (e.g., '90 min', '2 Seasons')
        
    Returns:
        tuple: (duration_value, duration_type)
    """
    if pd.isna(duration_string) or duration_string is None:
        return (0, None)
    
    parts = duration_string.split()
    if len(parts) != 2:
        return (0, None)
    
    try:
        value = int(parts[0])
        type_str = parts[1]
        return (value, type_str)
    except:
        return (0, None)

def plot_distribution(df, column, title=None, figsize=(10, 6)):
    """
    Plot distribution of values in a column.
    
    Args:
        df (DataFrame): Pandas DataFrame
        column (str): Column name
        title (str, optional): Plot title
        figsize (tuple, optional): Figure size
        
    Returns:
        matplotlib.figure.Figure: Plot figure
    """
    plt.figure(figsize=figsize)
    
    if df[column].dtype == 'object':
        # For categorical data
        value_counts = df[column].value_counts().sort_values(ascending=False)
        sns.barplot(x=value_counts.index[:15], y=value_counts.values[:15])
        plt.xticks(rotation=45, ha='right')
    else:
        # For numerical data
        sns.histplot(df[column].dropna())
    
    if title:
        plt.title(title)
    else:
        plt.title(f'Distribution of {column}')
    
    plt.tight_layout()
    return plt.gcf()

def generate_wordcloud(text_series, stopwords=None, figsize=(12, 8)):
    """
    Generate word cloud from a series of text.
    
    Args:
        text_series (Series): Pandas Series of text
        stopwords (set, optional): Set of stopwords to exclude
        figsize (tuple, optional): Figure size
        
    Returns:
        matplotlib.figure.Figure: Word cloud figure
    """
    text = ' '.join(text_series.dropna().astype(str))
    
    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        max_words=200,
        stopwords=stopwords,
        contour_width=3,
        contour_color='steelblue'
    ).generate(text)
    
    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    return plt.gcf()

def get_top_genres(df, n=10):
    """
    Get top N genres from the dataset.
    
    Args:
        df (DataFrame): DataFrame with 'listed_in' column
        n (int, optional): Number of top genres to return
        
    Returns:
        list: List of top genres
    """
    # Extract all genres
    all_genres = []
    for genres in df['listed_in'].dropna():
        all_genres.extend(split_genres(genres))
    
    # Count genres
    from collections import Counter
    genre_counts = Counter(all_genres)
    
    # Return top N
    return [genre for genre, count in genre_counts.most_common(n)]

def create_user_profile(watched_titles, df):
    """
    Create a user profile based on watched titles.
    
    Args:
        watched_titles (list): List of titles watched by the user
        df (DataFrame): DataFrame with movie/show information
        
    Returns:
        dict: User profile with genre preferences
    """
    watched_df = df[df['title'].isin(watched_titles)]
    
    # Extract all genres from watched titles
    genre_lists = watched_df['listed_in'].apply(split_genres)
    all_genres = [genre for sublist in genre_lists for genre in sublist]
    
    # Count genres
    from collections import Counter
    genre_counts = Counter(all_genres)
    
    # Normalize to get preferences
    total = sum(genre_counts.values())
    preferences = {genre: count/total for genre, count in genre_counts.items()}
    
    return {
        'watched_count': len(watched_titles),
        'genre_preferences': preferences,
        'country_preferences': watched_df['country'].value_counts().to_dict(),
        'type_preferences': watched_df['type'].value_counts().to_dict()
    }
