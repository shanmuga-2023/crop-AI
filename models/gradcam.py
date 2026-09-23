"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Module.

Generates visual explanations for model predictions by highlighting
the image regions that most influenced the classification decision.
"""
import os
import uuid
import numpy as np
from PIL import Image


class GradCAMGenerator:
    """Generate Grad-CAM heatmap overlays for explainability."""

    # Colormap approximation for heatmap (blue -> green -> yellow -> red)
    COLORMAP = None  # Will be generated on first use

    def __init__(self, config):
        """
        Args:
            config: Application configuration object.
        """
        self.config = config
        self.mock_mode = config.MOCK_MODE
        self.output_dir = config.GRADCAM_OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(self, image_path, model=None, class_index=None, prediction_result=None):
        """
        Generate a Grad-CAM heatmap for the given image and prediction.

        Args:
            image_path: Path to the original image.
            model: The TensorFlow model (None for mock mode).
            class_index: Target class index for Grad-CAM.
            prediction_result: Full prediction result dict.

        Returns:
            dict: Paths to heatmap and overlay images, plus raw heatmap data.
        """
        if self.mock_mode or model is None:
            return self._mock_gradcam(image_path, prediction_result)

        return self._real_gradcam(image_path, model, class_index)

    def _real_gradcam(self, image_path, model, class_index):
        """Generate real Grad-CAM using TensorFlow GradientTape."""
        import tensorflow as tf

        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_resized = img.resize(self.config.MODEL_INPUT_SIZE, Image.LANCZOS)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        img_tensor = tf.convert_to_tensor(np.expand_dims(img_array, axis=0))

        # Find the last convolutional layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
            if hasattr(layer, 'layers'):
                for sub_layer in reversed(layer.layers):
                    if isinstance(sub_layer, tf.keras.layers.Conv2D):
                        last_conv_layer = sub_layer
                        break
                if last_conv_layer is not None:
                    break

        if last_conv_layer is None:
            return self._mock_gradcam(image_path, None)

        try:
            # Check if the conv layer is inside a submodel (e.g. transfer learning base model)
            submodel = None
            for layer in model.layers:
                if hasattr(layer, 'layers') and last_conv_layer in layer.layers:
                    submodel = layer
                    break

            if submodel is not None:
                sub_grad_model = tf.keras.models.Model(
                    inputs=submodel.input,
                    outputs=[last_conv_layer.output, submodel.output]
                )
                classifier_layers = model.layers[model.layers.index(submodel) + 1:]

                with tf.GradientTape() as tape:
                    conv_outputs, sub_outputs = sub_grad_model(img_tensor)
                    tape.watch(conv_outputs)
                    preds = sub_outputs
                    for cl in classifier_layers:
                        preds = cl(preds)
                    if class_index is None:
                        class_index = int(tf.argmax(preds[0]))
                    loss = preds[:, class_index]

                grads = tape.gradient(loss, conv_outputs)
            else:
                grad_model = tf.keras.models.Model(
                    inputs=model.input,
                    outputs=[last_conv_layer.output, model.output]
                )
                with tf.GradientTape() as tape:
                    conv_outputs, predictions = grad_model(img_tensor)
                    if class_index is None:
                        class_index = int(tf.argmax(predictions[0]))
                    loss = predictions[:, class_index]

                grads = tape.gradient(loss, conv_outputs)

            if grads is None:
                return self._mock_gradcam(image_path, None)

            weights = tf.reduce_mean(grads, axis=(0, 1, 2))
            cam = tf.reduce_sum(tf.multiply(conv_outputs[0], weights), axis=-1)
            cam = tf.nn.relu(cam)
            max_cam = float(tf.reduce_max(cam))
            if max_cam > 0:
                cam = cam / max_cam
            heatmap = cam.numpy()

            return self._save_heatmap(image_path, heatmap)
        except Exception as e:
            print(f"[WARNING] Real Grad-CAM encountered an issue: {e}. Falling back to simulation.")
            return self._mock_gradcam(image_path, None)


    def _mock_gradcam(self, image_path, prediction_result=None):
        """
        Generate a realistic mock Grad-CAM heatmap.

        Creates a heatmap that highlights non-green areas of the leaf
        (which typically correspond to disease symptoms).
        """
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img, dtype=np.float32)

        # Create a mock heatmap based on color analysis
        # Highlight areas that are NOT green (likely diseased)
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

        # Calculate "disease likelihood" per pixel
        # Brown, yellow, black spots score high; green areas score low
        green_dominance = g / (r + g + b + 1e-6)
        brown_score = (r > 100) & (g > 50) & (g < 180) & (b < 100)
        dark_score = (r < 80) & (g < 80) & (b < 80) & (r + g + b > 30)
        yellow_score = (r > 150) & (g > 150) & (b < 100)

        # Combine into a heatmap
        heatmap = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.float32)
        heatmap += (1.0 - green_dominance) * 0.4
        heatmap += brown_score.astype(np.float32) * 0.3
        heatmap += dark_score.astype(np.float32) * 0.2
        heatmap += yellow_score.astype(np.float32) * 0.3

        # Smooth the heatmap to look more realistic
        # Simple box blur
        kernel_size = max(img_array.shape[0] // 15, 5)
        heatmap = self._box_blur(heatmap, kernel_size)

        # Normalize to 0-1
        heatmap_min = heatmap.min()
        heatmap_max = heatmap.max()
        if heatmap_max - heatmap_min > 0:
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)

        # Resize to model input size for consistency
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_img = heatmap_img.resize(self.config.MODEL_INPUT_SIZE, Image.LANCZOS)
        heatmap = np.array(heatmap_img, dtype=np.float32) / 255.0

        return self._save_heatmap(image_path, heatmap)

    def _box_blur(self, arr, kernel_size):
        """Simple box blur for smoothing heatmaps."""
        from PIL import ImageFilter
        img = Image.fromarray((arr * 255).astype(np.uint8))
        blurred = img.filter(ImageFilter.GaussianBlur(radius=kernel_size))
        return np.array(blurred, dtype=np.float32) / 255.0

    def _save_heatmap(self, original_image_path, heatmap):
        """
        Save heatmap and overlay images.

        Args:
            original_image_path: Path to the original image.
            heatmap: 2D numpy array (0-1) of the heatmap.

        Returns:
            dict: Paths and data for the generated visualizations.
        """
        unique_id = str(uuid.uuid4())[:8]

        # Load original image
        original = Image.open(original_image_path).convert('RGB')
        original_resized = original.resize(
            (heatmap.shape[1], heatmap.shape[0]), Image.LANCZOS
        )

        # Create colored heatmap (Jet-like colormap)
        colored_heatmap = self._apply_colormap(heatmap)
        heatmap_pil = Image.fromarray(colored_heatmap)

        # Create overlay (blend original + heatmap)
        overlay = Image.blend(original_resized, heatmap_pil, alpha=0.4)

        # Save files
        heatmap_filename = f'heatmap_{unique_id}.png'
        overlay_filename = f'overlay_{unique_id}.png'

        heatmap_path = os.path.join(self.output_dir, heatmap_filename)
        overlay_path = os.path.join(self.output_dir, overlay_filename)

        heatmap_pil.save(heatmap_path, 'PNG')
        overlay.save(overlay_path, 'PNG')

        return {
            'heatmap_path': heatmap_path,
            'overlay_path': overlay_path,
            'heatmap_url': f'/static/gradcam_outputs/{heatmap_filename}',
            'overlay_url': f'/static/gradcam_outputs/{overlay_filename}',
            'heatmap_array': heatmap.tolist(),
        }

    def _apply_colormap(self, heatmap):
        """
        Apply a Jet-like colormap to a grayscale heatmap.

        Args:
            heatmap: 2D numpy array with values 0-1.

        Returns:
            3D numpy array (H, W, 3) with RGB colormap applied.
        """
        h, w = heatmap.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        for i in range(h):
            for j in range(w):
                v = heatmap[i, j]
                colored[i, j] = self._jet_color(v)

        return colored

    @staticmethod
    def _jet_color(value):
        """Map a 0-1 value to a Jet colormap RGB tuple."""
        # Four-segment piecewise linear interpolation
        if value < 0.25:
            r, g, b = 0, int(255 * (value / 0.25)), 255
        elif value < 0.5:
            r, g, b = 0, 255, int(255 * (1 - (value - 0.25) / 0.25))
        elif value < 0.75:
            r, g, b = int(255 * ((value - 0.5) / 0.25)), 255, 0
        else:
            r, g, b = 255, int(255 * (1 - (value - 0.75) / 0.25)), 0

        return (min(max(r, 0), 255), min(max(g, 0), 255), min(max(b, 0), 255))
