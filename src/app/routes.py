from flask import Blueprint, request, jsonify, current_app
import pandas as pd
from .recommendation_engine import RecommendationEngine
from .user_manager import UserManager

# Create blueprint
main_bp = Blueprint('main', __name__)

# Initialize components (these would be properly initialized in a real app)
rec_engine = None
user_manager = None

@main_bp.before_app_first_request
def initialize_components():
    """Initialize recommendation engine and user manager on first request"""
    global rec_engine, user_manager
    
    # Load processed data
    from src.data.loader import load_netflix_data
    df = load_netflix_data('processed/netflix_processed.csv')
    
    # Initialize components
    rec_engine = RecommendationEngine(df)
    user_manager = UserManager()

@main_bp.route('/api/survey', methods=['GET'])
def get_survey_titles():
    """Get a list of popular titles for the initial survey"""
    # Return a sample of diverse and popular content
    sample_titles = rec_engine.get_diverse_titles(n=50)
    return jsonify({
        'success': True,
        'titles': sample_titles
    })

@main_bp.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Get personalized recommendations based on liked content"""
    data = request.json
    user_id = data.get('user_id', 'anonymous')
    liked_titles = data.get('liked_titles', [])
    
    # Store user preferences
    if liked_titles:
        user_manager.update_preferences(user_id, liked_titles)
    
    # Get personalized recommendations
    recommendations = rec_engine.recommend_for_user(
        user_id=user_id,
        liked_titles=liked_titles,
        n=10
    )
    
    return jsonify({
        'success': True,
        'recommendations': recommendations
    })

@main_bp.route('/api/title/<title>', methods=['GET'])
def get_title_recommendations(title):
    """Get similar content to a specific title"""
    try:
        n = int(request.args.get('n', 5))
        similar_titles = rec_engine.recommend_similar(title, n=n)
        
        return jsonify({
            'success': True,
            'title': title,
            'similar': similar_titles
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 404

@main_bp.route('/api/user/<user_id>/profile', methods=['GET'])
def get_user_profile(user_id):
    """Get a user's preference profile"""
    profile = user_manager.get_profile(user_id)
    return jsonify({
        'success': True,
        'profile': profile
    })

@main_bp.route('/api/user/<user_id>/rate', methods=['POST'])
def rate_title(user_id):
    """Store a user's rating for a title"""
    data = request.json
    title = data.get('title')
    rating = data.get('rating')  # 1-5 scale
    
    if not title or rating is None:
        return jsonify({
            'success': False,
            'error': 'Missing required data'
        }), 400
    
    user_manager.add_rating(user_id, title, rating)
    
    # Get new recommendations based on updated preferences
    recommendations = rec_engine.recommend_for_user(user_id, n=5)
    
    return jsonify({
        'success': True,
        'recommendations': recommendations
    })
