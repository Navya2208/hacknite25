import json
import os
import time
from datetime import datetime
import pandas as pd

class UserManager:
    """Manages user preferences, ratings, and viewing history"""
    
    def __init__(self, data_dir='data/users'):
        """Initialize with a directory to store user data"""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def _get_user_file(self, user_id):
        """Get the file path for a user's data"""
        return os.path.join(self.data_dir, f"{user_id}.json")
    
    def _load_user_data(self, user_id):
        """Load a user's data from file"""
        file_path = self._get_user_file(user_id)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return self._create_default_profile()
    
    def _save_user_data(self, user_id, data):
        """Save a user's data to file"""
        file_path = self._get_user_file(user_id)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _create_default_profile(self):
        """Create a default user profile"""
        return {
            'liked_titles': [],
            'ratings': {},  # title -> rating (1-5)
            'watch_history': [],  # list of {title, timestamp}
            'genre_preferences': {},  # genre -> weight (0-1)
            'last_updated': datetime.now().isoformat()
        }
    
    def get_profile(self, user_id):
        """Get a user's preference profile"""
        return self._load_user_data(user_id)
    
    def update_preferences(self, user_id, liked_titles):
        """Update a user's liked titles and genre preferences"""
        user_data = self._load_user_data(user_id)
        
        # Update liked titles (add new ones without duplicates)
        current_liked = set(user_data['liked_titles'])
        current_liked.update(liked_titles)
        user_data['liked_titles'] = list(current_liked)
        
        # Update genre preferences based on liked titles
        self._update_genre_preferences(user_id, user_data)
        
        # Save updated data
        user_data['last_updated'] = datetime.now().isoformat()
        self._save_user_data(user_id, user_data)
        
        return user_data
    
    def add_rating(self, user_id, title, rating):
        """Add or update a user's rating for a title"""
        user_data = self._load_user_data(user_id)
        
        # Update rating
        user_data['ratings'][title] = rating
        
        # Add to watch history if not already there
        if title not in [entry['title'] for entry in user_data['watch_history']]:
            user_data['watch_history'].append({
                'title': title,
                'timestamp': datetime.now().isoformat()
            })
        
        # Update genre preferences
        self._update_genre_preferences(user_id, user_data)
        
        # Save updated data
        user_data['last_updated'] = datetime.now().isoformat()
        self._save_user_data(user_id, user_data)
        
        return user_data
    
    def add_to_watch_history(self, user_id, title):
        """Add a title to the user's watch history"""
        user_data = self._load_user_data(user_id)
        
        # Add to watch history with timestamp
        user_data['watch_history'].append({
            'title': title,
            'timestamp': datetime.now().isoformat()
        })
        
        # Save updated data
        user_data['last_updated'] = datetime.now().isoformat()
        self._save_user_data(user_id, user_data)
        
        return user_data
    
    def _update_genre_preferences(self, user_id, user_data=None):
        """Update genre preferences based on liked titles and ratings"""
        if user_data is None:
            user_data = self._load_user_data(user_id)
        
        # For this method, we need access to the movie database
        # In a real app, this would be passed during initialization
        try:
            from .recommendation_engine import RecommendationEngine
            rec_engine = RecommendationEngine.instance()
            title_to_genres = rec_engine.title_to_genres
        except (ImportError, AttributeError):
            # Fallback if we can't access the recommendation engine
            # This is just for example - in a real app, you'd handle this better
            title_to_genres = {}
        
        # Count genres from liked titles and ratings
        genre_counts = {}
        
        # Process liked titles
        for title in user_data['liked_titles']:
            if title in title_to_genres:
                for genre in title_to_genres[title]:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        # Process ratings (weighted by rating value)
        for title, rating in user_data['ratings'].items():
            if title in title_to_genres:
                for genre in title_to_genres[title]:
                    # Weight by rating (1-5 scale)
                    genre_counts[genre] = genre_counts.get(genre, 0) + (rating / 5.0) 
        
        # Normalize to get preferences
        total = sum(genre_counts.values()) if genre_counts else 1
        user_data['genre_preferences'] = {genre: count/total for genre, count in genre_counts.items()}
        
        return user_data
    
    def get_recommendations(self, user_id, n=10):
        """Get recommendations based on user profile (delegate to recommendation engine)"""
        # This is just a convenience method that delegates to the recommendation engine
        from .recommendation_engine import RecommendationEngine
        rec_engine = RecommendationEngine.instance()
        return rec_engine.recommend_for_user(user_id, n=n)
