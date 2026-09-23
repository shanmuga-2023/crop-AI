"""
Disease Severity Estimation Module.

Analyzes leaf images using color segmentation to estimate the
percentage of leaf area affected by disease.
"""
import numpy as np
from PIL import Image


class SeverityEstimator:
    """Estimate disease severity from leaf image analysis."""

    # Severity classification thresholds
    SEVERITY_LEVELS = {
        'healthy': {'min': 0, 'max': 5, 'color': '#10b981', 'icon': '✅'},
        'mild': {'min': 5, 'max': 20, 'color': '#22d3ee', 'icon': '🔵'},
        'moderate': {'min': 20, 'max': 40, 'color': '#f59e0b', 'icon': '🟡'},
        'severe': {'min': 40, 'max': 60, 'color': '#f97316', 'icon': '🟠'},
        'critical': {'min': 60, 'max': 100, 'color': '#ef4444', 'icon': '🔴'},
    }

    def estimate(self, image_path):
        """
        Estimate disease severity from a leaf image.

        Uses HSV color space segmentation to:
        1. Separate the leaf from the background
        2. Identify diseased regions within the leaf
        3. Calculate the severity percentage

        Args:
            image_path: Path to the leaf image.

        Returns:
            dict: Severity data including percentage, level, and analysis details.
        """
        # Load image
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img, dtype=np.float32)

        # Convert to HSV-like representation
        hsv = self._rgb_to_hsv(img_array / 255.0)

        # Step 1: Segment the leaf from background
        leaf_mask = self._segment_leaf(img_array, hsv)
        total_leaf_pixels = np.sum(leaf_mask)

        if total_leaf_pixels < 100:
            # Too few leaf pixels detected — probably an issue with segmentation
            return self._fallback_estimate(img_array)

        # Step 2: Segment diseased areas within the leaf
        disease_mask = self._segment_disease(img_array, hsv, leaf_mask)
        diseased_pixels = np.sum(disease_mask)

        # Step 3: Calculate severity percentage
        severity_pct = (diseased_pixels / total_leaf_pixels) * 100
        severity_pct = min(max(severity_pct, 0), 100)

        # Classify severity level
        severity_level = self._classify_severity(severity_pct)

        return {
            'severity_percentage': round(float(severity_pct), 2),
            'severity_level': severity_level,
            'severity_info': self.SEVERITY_LEVELS[severity_level],
            'analysis': {
                'total_leaf_pixels': int(total_leaf_pixels),
                'diseased_pixels': int(diseased_pixels),
                'healthy_pixels': int(total_leaf_pixels - diseased_pixels),
                'image_size': f'{img_array.shape[1]}x{img_array.shape[0]}',
                'total_image_pixels': int(img_array.shape[0] * img_array.shape[1]),
                'leaf_coverage_pct': round(
                    float(total_leaf_pixels / (img_array.shape[0] * img_array.shape[1])) * 100, 1
                ),
            },
        }

    def _rgb_to_hsv(self, rgb):
        """
        Convert RGB (0-1 range) to HSV.

        Returns array with H (0-360), S (0-1), V (0-1).
        """
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        delta = maxc - minc

        # Value
        v = maxc

        # Saturation
        s = np.where(maxc > 0, delta / (maxc + 1e-10), 0)

        # Hue
        h = np.zeros_like(maxc)
        mask_r = (maxc == r) & (delta > 0)
        mask_g = (maxc == g) & (delta > 0)
        mask_b = (maxc == b) & (delta > 0)

        h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / (delta[mask_r] + 1e-10)) % 6)
        h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / (delta[mask_g] + 1e-10)) + 2)
        h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / (delta[mask_b] + 1e-10)) + 4)

        return np.stack([h, s, v], axis=-1)

    def _segment_leaf(self, rgb, hsv):
        """
        Segment the leaf from the background.

        Detects pixels that are part of the leaf (both healthy green
        and diseased areas) vs. the background.
        """
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Green leaf pixels (hue 35-150, moderate to high saturation)
        green_mask = (h >= 35) & (h <= 150) & (s >= 0.1) & (v >= 0.1)

        # Brown/yellow diseased pixels (hue 0-35 or 150-50, still on leaf)
        brown_mask = (
            ((h >= 0) & (h < 35) | (h > 150) & (h < 200)) &
            (s >= 0.1) & (v >= 0.15) & (v <= 0.95)
        )

        # Dark spots on leaf (low value but not background)
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        dark_mask = (v >= 0.05) & (v < 0.3) & (s >= 0.05)

        # Very bright areas are likely background
        bright_bg = (v > 0.95) & (s < 0.1)

        # Very dark areas might be background
        very_dark_bg = (v < 0.05)

        # Combine leaf masks
        leaf_mask = (green_mask | brown_mask | dark_mask) & ~bright_bg & ~very_dark_bg

        return leaf_mask

    def _segment_disease(self, rgb, hsv, leaf_mask):
        """
        Segment diseased areas within the leaf.

        Identifies brown spots, yellow chlorosis, black necrosis,
        and other non-green areas within the leaf boundary.
        """
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        # Brown/tan lesions
        brown = (
            ((h >= 0) & (h < 40)) &
            (s >= 0.15) &
            (v >= 0.15) & (v < 0.85) &
            (r > g)
        )

        # Yellow chlorosis
        yellow = (
            (h >= 30) & (h < 65) &
            (s >= 0.3) &
            (v >= 0.5) &
            (r > 130) & (g > 130) & (b < 100)
        )

        # Dark necrotic spots
        necrotic = (
            (v < 0.25) &
            (s < 0.3) &
            (v > 0.03)
        )

        # Reddish-purple spots (some diseases)
        reddish = (
            ((h >= 300) | (h < 15)) &
            (s >= 0.2) &
            (v >= 0.15) &
            (r > g) & (r > b)
        )

        # White powdery mildew
        white_powder = (
            (v > 0.8) &
            (s < 0.15) &
            (h >= 0) & (h < 360)
        )

        # Combine disease masks, only within leaf boundary
        disease_mask = (brown | yellow | necrotic | reddish | white_powder) & leaf_mask

        # Exclude healthy green areas that might overlap
        healthy_green = (h >= 60) & (h <= 140) & (s >= 0.2) & (v >= 0.2)
        disease_mask = disease_mask & ~healthy_green

        return disease_mask

    def _classify_severity(self, severity_pct):
        """Classify a severity percentage into a named level."""
        for level, data in self.SEVERITY_LEVELS.items():
            if data['min'] <= severity_pct < data['max']:
                return level
        return 'critical'

    def _fallback_estimate(self, img_array):
        """
        Fallback estimation when leaf segmentation fails.

        Uses overall image color distribution as a rough proxy.
        """
        r = np.mean(img_array[:, :, 0])
        g = np.mean(img_array[:, :, 1])
        b = np.mean(img_array[:, :, 2])

        # Simple heuristic: more red/brown = more diseased
        green_ratio = g / (r + g + b + 1e-6)
        estimated_pct = max(0, min(100, (1 - green_ratio) * 100 * 0.5))

        severity_level = self._classify_severity(estimated_pct)

        return {
            'severity_percentage': round(float(estimated_pct), 2),
            'severity_level': severity_level,
            'severity_info': self.SEVERITY_LEVELS[severity_level],
            'analysis': {
                'total_leaf_pixels': 0,
                'diseased_pixels': 0,
                'healthy_pixels': 0,
                'image_size': f'{img_array.shape[1]}x{img_array.shape[0]}',
                'total_image_pixels': img_array.shape[0] * img_array.shape[1],
                'leaf_coverage_pct': 0,
                'note': 'Fallback estimation — leaf segmentation was insufficient',
            },
        }
