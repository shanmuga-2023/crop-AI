"""
Crop Disease Expert System — Flask Application.

Main entry point for the web application. Initializes all components
and registers route blueprints.
"""
import os
from flask import Flask, send_from_directory
from flask_cors import CORS
from config import Config


def create_app():
    """Application factory."""
    app = Flask(__name__)
    app.config.from_object(Config)
    Config.init_app(app)
    CORS(app)

    # Initialize core components and store in app context
    from models.classifier import DiseaseClassifier
    from models.gradcam import GradCAMGenerator
    from models.severity import SeverityEstimator
    from expert_system.knowledge_base import KnowledgeBase
    from expert_system.risk_calculator import RiskCalculator
    from database.db import Database

    app.classifier = DiseaseClassifier(Config)
    app.gradcam = GradCAMGenerator(Config)
    app.severity_estimator = SeverityEstimator()
    app.knowledge_base = KnowledgeBase(Config.KNOWLEDGE_BASE_PATH)
    app.risk_calculator = RiskCalculator(app.knowledge_base)
    app.db = Database(Config.DATABASE_PATH)

    # Register blueprints
    from routes.diagnosis import diagnosis_bp
    from routes.whatif import whatif_bp
    from routes.history import history_bp

    app.register_blueprint(diagnosis_bp)
    app.register_blueprint(whatif_bp)
    app.register_blueprint(history_bp)

    # --- Page routes ---

    @app.route('/')
    def landing_page():
        """Serve the landing page."""
        from flask import render_template
        return render_template('landing.html')

    @app.route('/diagnose')
    def diagnose_page():
        """Serve the main diagnosis application page."""
        from flask import render_template
        crops = app.knowledge_base.get_all_crop_names()
        growth_stages = app.knowledge_base.growth_stages
        return render_template('index.html', crops=crops, growth_stages=growth_stages)

    @app.route('/history')
    def history_page():
        """Serve the diagnosis history page."""
        from flask import render_template
        return render_template('history.html')

    @app.route('/uploads/<filename>')
    def uploaded_file(filename):
        """Serve uploaded images."""
        return send_from_directory(Config.UPLOAD_FOLDER, filename)

    @app.route('/api/model-info')
    def model_info():
        """Return model information."""
        from flask import jsonify
        return jsonify(app.classifier.get_model_info())

    @app.route('/api/crops')
    def get_crops():
        """Return available crop names."""
        from flask import jsonify
        return jsonify({
            'crops': app.knowledge_base.get_all_crop_names(),
            'growth_stages': app.knowledge_base.growth_stages,
        })

    return app


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=port)

