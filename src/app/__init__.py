from flask import Flask
from flask_cors import CORS

def create_app(config=None):
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Enable CORS for all routes
    CORS(app)
    
    # Load default configuration
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE_URI='sqlite:///netflix_recommendations.db',
        CACHE_TYPE='simple',
        DEBUG=True
    )
    
    # Override with custom config if provided
    if config:
        app.config.update(config)
    
    # Register blueprints
    from .routes import main_bp
    app.register_blueprint(main_bp)
    
    return app
