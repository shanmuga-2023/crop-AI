"""
Configuration settings for the Crop Disease Expert System.
"""
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'crop-disease-expert-system-secret-key')

    # Upload settings
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

    # Database
    DATABASE_PATH = os.path.join(BASE_DIR, 'database', 'diagnoses.db')

    # Model settings
    MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, 'models', 'weights', 'resnet50_plantvillage.h5')
    MODEL_INPUT_SIZE = (224, 224)
    NUM_CLASSES = 38
    MOCK_MODE = not os.path.exists(
        os.path.join(BASE_DIR, 'models', 'weights', 'resnet50_plantvillage.h5')
    )

    # Knowledge base
    KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, 'expert_system', 'knowledge', 'diseases.json')

    # Grad-CAM output
    GRADCAM_OUTPUT_DIR = os.path.join(BASE_DIR, 'static', 'gradcam_outputs')

    # PlantVillage class labels (38 classes)
    CLASS_LABELS = [
        'Apple___Apple_scab',
        'Apple___Black_rot',
        'Apple___Cedar_apple_rust',
        'Apple___healthy',
        'Blueberry___healthy',
        'Cherry_(including_sour)___Powdery_mildew',
        'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot',
        'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)___healthy',
        'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)',
        'Peach___Bacterial_spot',
        'Peach___healthy',
        'Pepper,_bell___Bacterial_spot',
        'Pepper,_bell___healthy',
        'Potato___Early_blight',
        'Potato___Late_blight',
        'Potato___healthy',
        'Raspberry___healthy',
        'Soybean___healthy',
        'Squash___Powdery_mildew',
        'Strawberry___Leaf_scorch',
        'Strawberry___healthy',
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites_Two-spotted_spider_mite',
        'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy',
    ]

    @staticmethod
    def init_app(app):
        """Initialize application directories."""
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.GRADCAM_OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(Config.DATABASE_PATH), exist_ok=True)
        os.makedirs(os.path.join(BASE_DIR, 'models', 'weights'), exist_ok=True)
