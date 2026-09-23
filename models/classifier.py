"""
Disease Classification Model.

Loads a ResNet50 model fine-tuned on PlantVillage dataset (38 classes).
Falls back to mock predictions when model weights are unavailable.
"""
import os
import numpy as np
from PIL import Image

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DiseaseClassifier:
    """Crop disease classification using ResNet50 transfer learning."""

    def __init__(self, config):
        """
        Initialize the classifier.

        Args:
            config: Application configuration object.
        """
        self.config = config
        self.model = None
        self.mock_mode = config.MOCK_MODE
        self.class_labels = list(config.CLASS_LABELS)
        self.input_size = config.MODEL_INPUT_SIZE

        # Check for custom class labels file alongside weights
        classes_path = os.path.join(os.path.dirname(config.MODEL_WEIGHTS_PATH), 'classes.json')
        if os.path.exists(classes_path):
            try:
                import json
                with open(classes_path, 'r') as f:
                    self.class_labels = json.load(f)
            except Exception as e:
                print(f"[WARNING] Failed to load classes.json: {e}")

        if not self.mock_mode:
            self._load_model()
        else:
            print("[INFO] Running in MOCK MODE — no model weights found.")
            print(f"[INFO] Expected weights at: {config.MODEL_WEIGHTS_PATH}")

    def _load_model(self):
        """Load the trained ResNet50 model."""
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.config.MODEL_WEIGHTS_PATH)
            print(f"[INFO] Model loaded successfully from {self.config.MODEL_WEIGHTS_PATH}")
        except Exception as e:
            print(f"[WARNING] Failed to load model: {e}")
            print("[INFO] Falling back to MOCK MODE.")
            self.mock_mode = True

    def preprocess_image(self, image_path):
        """
        Preprocess an image for model input.

        Args:
            image_path: Path to the image file.

        Returns:
            numpy array: Preprocessed image tensor of shape (1, 224, 224, 3).
        """
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.input_size, Image.LANCZOS)
        img_array = np.array(img, dtype=np.float32)

        # ResNet50 preprocessing (ImageNet normalization)
        img_array = img_array / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std

        return np.expand_dims(img_array, axis=0)

    def predict(self, image_path, top_k=3):
        """
        Predict disease from a leaf image.

        Args:
            image_path: Path to the uploaded leaf image.
            top_k: Number of top predictions to return.

        Returns:
            dict: Prediction results with top-k labels and confidence scores.
        """
        if self.mock_mode:
            return self._mock_predict(image_path, top_k)

        # Preprocess
        img_tensor = self.preprocess_image(image_path)

        # Predict
        import tensorflow as tf
        predictions = self.model.predict(img_tensor, verbose=0)[0]

        # Get top-k
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            results.append({
                'class_label': self.class_labels[idx],
                'class_index': int(idx),
                'confidence': float(predictions[idx]),
            })

        return {
            'top_prediction': results[0],
            'top_k_predictions': results,
            'all_probabilities': predictions.tolist(),
            'mock_mode': False,
        }

    def _mock_predict(self, image_path, top_k=3):
        """
        Generate deterministic mock predictions for demonstration.

        Uses image characteristics (file size, average color) to produce
        consistent, plausible predictions without a trained model.
        """
        # Use image properties for deterministic mock results
        img = Image.open(image_path).convert('RGB')
        img_small = img.resize((32, 32))
        pixels = np.array(img_small, dtype=np.float32)

        # Use average color channels as a seed for deterministic selection
        avg_r = np.mean(pixels[:, :, 0])
        avg_g = np.mean(pixels[:, :, 1])
        avg_b = np.mean(pixels[:, :, 2])

        # Compute a hash-like index from color values
        color_hash = int(avg_r * 100 + avg_g * 10 + avg_b) % len(self.class_labels)

        # Determine if leaf looks "healthy" based on green dominance
        green_ratio = avg_g / (avg_r + avg_g + avg_b + 1e-6)
        is_greenish = green_ratio > 0.38

        if is_greenish:
            # Prefer healthy classes
            healthy_classes = [i for i, l in enumerate(self.class_labels) if 'healthy' in l]
            primary_idx = healthy_classes[color_hash % len(healthy_classes)]
            primary_confidence = 0.82 + (avg_g % 10) * 0.015
        else:
            # Prefer diseased classes
            disease_classes = [i for i, l in enumerate(self.class_labels) if 'healthy' not in l]
            primary_idx = disease_classes[color_hash % len(disease_classes)]
            primary_confidence = 0.72 + (avg_r % 10) * 0.02

        primary_confidence = min(primary_confidence, 0.97)

        # Generate secondary predictions
        remaining_confidence = 1.0 - primary_confidence
        secondary_indices = []
        for offset in range(1, top_k):
            sec_idx = (primary_idx + offset * 7) % len(self.class_labels)
            if sec_idx == primary_idx:
                sec_idx = (sec_idx + 1) % len(self.class_labels)
            secondary_indices.append(sec_idx)

        results = [{
            'class_label': self.class_labels[primary_idx],
            'class_index': primary_idx,
            'confidence': round(primary_confidence, 4),
        }]

        for i, sec_idx in enumerate(secondary_indices):
            sec_conf = remaining_confidence * (0.6 if i == 0 else 0.4)
            results.append({
                'class_label': self.class_labels[sec_idx],
                'class_index': sec_idx,
                'confidence': round(sec_conf, 4),
            })

        # Generate full probability array
        all_probs = [0.0] * len(self.class_labels)
        for r in results:
            all_probs[r['class_index']] = r['confidence']

        return {
            'top_prediction': results[0],
            'top_k_predictions': results,
            'all_probabilities': all_probs,
            'mock_mode': True,
        }

    def get_model_info(self):
        """Get information about the loaded model."""
        return {
            'model_type': 'ResNet50 (Transfer Learning)',
            'dataset': 'PlantVillage',
            'num_classes': len(self.class_labels),
            'input_size': f'{self.input_size[0]}x{self.input_size[1]}',
            'mock_mode': self.mock_mode,
        }
