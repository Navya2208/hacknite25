import pandas as pd
import numpy as np
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative_filtering import CollaborativeFilteringRecommender
from src.utils.helpers import split_genres, create_user_profile

class RecommendationEngine:
    """Central recommendation engine that combines different recommendation strategies"""
    
    def __init__(self, df):
        """Initialize with preprocessed Netflix data"""
        self.df = df
        self.content_model = ContentBasedRecommender().fit(df)
        self.collab_model = None  # Will be initialized when we have user ratings
        
        # Map titles to indices for quick lookup
        self.title_to_idx = {title.lower(): i for i, title in enumerate(df['title'])}
        
        # Extract genre info for later use
        self.title_to_genres = {
            row['title']: split_genres(row['listed_in']) 
            for _, row in df.iterrows()
        }
    
    def get_diverse_titles(self, n=50):
        """Get a diverse sample of titles for the initial survey"""
        # Get popular titles across different genres
        genres = self.df['listed_in'].apply(lambda x: x.split(',')[0].strip()).unique()
        
        sample_titles = []
        titles_per_genre = max(2, n // len(genres))
        
        for genre in genres:
            genre_df = self.df[self.df['listed_in'].str.contains(genre)]
            if len(genre_df) > 0:
                # Get a mix of movies and shows for each genre
                for type_val in ['Movie', 'TV Show']:
                    type_df = genre_df[genre_df['type'] == type_val]
                    if len(type_df) > 0:
                        sample = type_df.sample(min(titles_per_genre // 2, len(type_df)))
                        for _, row in sample.iterrows():
                            sample_titles.append({
                                'id': row['show_id'],
                                'title': row['title'],
                                'type': row['type'],
                                'genre': genre,
                                'description': row['description'][:100] + '...' if len(row['description']) > 100 else row['description']
                            })
        
        # If we don't have enough titles, add random ones
        if len(sample_titles) < n:
            remaining = n - len(sample_titles)
            remaining_df = self.df[~self.df['show_id'].isin([t['id'] for t in sample_titles])]
            if len(remaining_df) > 0:
                random_sample = remaining_df.sample(min(remaining, len(remaining_df)))
                for _, row in random_sample.iterrows():
                    genre = row['listed_in'].split(',')[0].strip() if not pd.isna(row['listed_in']) else 'Unknown'
                    sample_titles.append({
                        'id': row['show_id'],
                        'title': row['title'],
                        'type': row['type'],
                        'genre': genre,
                        'description': row['description'][:100] + '...' if len(row['description']) > 100 else row['description']
                    })
        
        return sample_titles[:n]
    
    def recommend_similar(self, title, n=5):
        """Recommend content similar to the given title"""
        try:
            recommendations = self.content_model.recommend(title, n=n)
            return self._format_recommendations(recommendations)
        except KeyError:
            # Title not found, return empty list
            return []
    
    def recommend_for_user(self, user_id, liked_titles=None, n=10):
        """Get personalized recommendations for a user"""
        from .user_manager import UserManager
        user_manager = UserManager()
        
        # If no liked titles provided, get from user history
        if not liked_titles:
            profile = user_manager.get_profile(user_id)
            liked_titles = profile.get('liked_titles', [])
        
        if not liked_titles:
            # Cold start: return diverse recommendations
            return self._format_recommendations(self.df.sample(n))
        
        # Content-based recommendations for each liked title
        all_recommendations = []
        for title in liked_titles:
            try:
                similar_titles = self.content_model.recommend(title, n=3)
                all_recommendations.append(similar_titles)
            except KeyError:
                # Title not found, skip
                continue
        
        if not all_recommendations:
            return self._format_recommendations(self.df.sample(n))
        
        # Combine all recommendations
        combined_df = pd.concat(all_recommendations)
        
        # Remove duplicates and already liked titles
        combined_df = combined_df.drop_duplicates(subset=['show_id'])
        combined_df = combined_df[~combined_df['title'].isin(liked_titles)]
        
        # Sort by relevance (if we have multiple recommendations)
        if len(combined_df) > n:
            # Get user genre preferences
            profile = user_manager.get_profile(user_id)
            genre_preferences = profile.get('genre_preferences', {})
            
            if genre_preferences:
                # Score recommendations by genre match
                combined_df['score'] = combined_df.apply(
                    lambda x: self._calculate_genre_score(x['listed_in'], genre_preferences),
                    axis=1
                )
                combined_df = combined_df.sort_values('score', ascending=False)
        
        # Return top N
        return self._format_recommendations(combined_df.head(n))
    
    def _calculate_genre_score(self, genres_str, preferences):
        """Calculate a score based on how well genres match user preferences"""
        if pd.isna(genres_str):
            return 0
            
        # Extract genres
        genres = split_genres(genres_str)
        
        # Calculate score
        score = 0
        for genre in genres:
            score += preferences.get(genre, 0)
            
        return score / len(genres) if genres else 0
    
    def _format_recommendations(self, recommendations_df):
        """Format recommendation DataFrame into a list of dictionaries"""
        return [
            {
                'id': row['show_id'],
                'title': row['title'],
                'type': row['type'],
                'description': row['description'],
                'genres': split_genres(row['listed_in']) if not pd.isna(row['listed_in']) else [],
                'year': int(row['release_year']) if not pd.isna(row['release_year']) else None,
                'duration': row['duration'],
                'rating': row['rating']
            }
            for _, row in recommendations_df.iterrows()
        ]
