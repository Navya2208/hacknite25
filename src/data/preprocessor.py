import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

def preprocess_netflix_df(df):
    """
    Cleans and preprocesses the Netflix DataFrame for ML/recommendation use.
    Returns the processed DataFrame.
    """
    # Fill missing values
    for col in ['director', 'cast', 'country', 'description']:
        df[col] = df[col].fillna('Unknown')
    
    # Standardize date_added and extract year/month
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['added_year'] = df['date_added'].dt.year
    df['added_month'] = df['date_added'].dt.month

    # Extract duration number and type
    df['duration_int'] = df['duration'].str.extract(r'(\d+)').astype(float)
    df['duration_type'] = df['duration'].str.extract(r'(min|Season|Seasons)')
    df['duration_int'] = df['duration_int'].fillna(0)
    df['duration_type'] = df['duration_type'].fillna('Unknown')

    # One-hot encode 'type'
    df = pd.get_dummies(df, columns=['type'])

    # Process genres (listed_in)
    df['genres'] = df['listed_in'].fillna('').apply(lambda x: [g.strip() for g in x.split(',')] if x else [])
    mlb = MultiLabelBinarizer()
    genre_dummies = pd.DataFrame(
        mlb.fit_transform(df['genres']),
        columns=mlb.classes_,
        index=df.index
    )
    df = pd.concat([df, genre_dummies], axis=1)

    # Lowercase and strip text fields
    for col in ['title', 'director', 'cast', 'country', 'description']:
        df[col] = df[col].astype(str).str.lower().str.strip()

    # Create a 'soup' column for content-based recommendation
    df['soup'] = (
        df['title'] + ' ' +
        df['director'] + ' ' +
        df['cast'] + ' ' +
        df['listed_in'].fillna('') + ' ' +
        df['description']
    ).str.replace(',', ' ')

    return df
