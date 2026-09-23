"""
Knowledge Base Loader and Manager.

Loads disease data from the JSON knowledge base and provides
query interfaces for the expert system engine.
"""
import json
import os


class KnowledgeBase:
    """Manages the disease knowledge base loaded from JSON."""

    def __init__(self, knowledge_path=None):
        """
        Initialize the knowledge base.

        Args:
            knowledge_path: Path to the diseases.json file.
                           Defaults to the bundled knowledge file.
        """
        if knowledge_path is None:
            knowledge_path = os.path.join(
                os.path.dirname(__file__), 'knowledge', 'diseases.json'
            )

        with open(knowledge_path, 'r') as f:
            self._data = json.load(f)

        self.diseases = self._data.get('diseases', {})
        self.healthy_classes = self._data.get('healthy_classes', [])
        self.crop_mapping = self._data.get('crop_mapping', {})
        self.growth_stages = self._data.get('growth_stages', [])
        self.environmental_factors = self._data.get('environmental_factors', {})

    def is_healthy(self, class_label):
        """Check if a predicted class represents a healthy plant."""
        return class_label in self.healthy_classes

    def get_disease_info(self, class_label):
        """
        Get full disease information for a given class label.

        Args:
            class_label: The PlantVillage class label (e.g., 'Tomato___Early_blight')

        Returns:
            dict: Disease information or None if not found.
        """
        return self.diseases.get(class_label, None)

    def get_crop_from_label(self, class_label):
        """Extract the crop name from a class label."""
        for crop, labels in self.crop_mapping.items():
            if class_label in labels:
                return crop
        # Fallback: extract from label
        parts = class_label.split('___')
        return parts[0].replace('_', ' ') if parts else 'Unknown'

    def get_diseases_for_crop(self, crop_name):
        """Get all disease labels associated with a crop."""
        return self.crop_mapping.get(crop_name, [])

    def get_symptoms(self, class_label):
        """Get symptom list for a disease."""
        info = self.get_disease_info(class_label)
        return info.get('symptoms', []) if info else []

    def get_causes(self, class_label):
        """Get causes for a disease."""
        info = self.get_disease_info(class_label)
        return info.get('causes', []) if info else []

    def get_treatment(self, class_label, treatment_type='both'):
        """
        Get treatment recommendations.

        Args:
            class_label: Disease class label.
            treatment_type: 'organic', 'chemical', or 'both'.

        Returns:
            dict: Treatment recommendations.
        """
        info = self.get_disease_info(class_label)
        if not info:
            return {}

        treatment = info.get('treatment', {})

        if treatment_type == 'organic':
            return {'organic': treatment.get('organic', [])}
        elif treatment_type == 'chemical':
            return {'chemical': treatment.get('chemical', [])}
        return treatment

    def get_preventive_measures(self, class_label):
        """Get preventive measures for a disease."""
        info = self.get_disease_info(class_label)
        return info.get('preventive_measures', []) if info else []

    def get_favorable_conditions(self, class_label):
        """Get conditions favorable for disease development."""
        info = self.get_disease_info(class_label)
        return info.get('favorable_conditions', {}) if info else {}

    def get_severity_thresholds(self, class_label):
        """Get severity classification thresholds for a disease."""
        info = self.get_disease_info(class_label)
        return info.get('severity_thresholds', {}) if info else {}

    def get_growth_stage_vulnerability(self, class_label):
        """Get vulnerability multipliers by growth stage."""
        info = self.get_disease_info(class_label)
        return info.get('growth_stage_vulnerability', {}) if info else {}

    def get_risk_modifiers(self, class_label):
        """Get risk modifiers (farming type, irrigation, etc.)."""
        info = self.get_disease_info(class_label)
        return info.get('risk_modifiers', {}) if info else {}

    def classify_humidity(self, humidity_value):
        """Classify a humidity percentage into a level category."""
        levels = self.environmental_factors.get('humidity_levels', {})
        for level_name, level_data in levels.items():
            if level_data['min'] <= humidity_value <= level_data['max']:
                return level_name, level_data['risk_factor']
        return 'moderate', 1.0

    def classify_temperature(self, temperature_value):
        """Classify a temperature value into a range category."""
        ranges = self.environmental_factors.get('temperature_ranges', {})
        for range_name, range_data in ranges.items():
            if range_data['min'] <= temperature_value < range_data['max']:
                return range_name, range_data['risk_factor']
        return 'warm', 1.2

    def get_rainfall_risk_factor(self, rainfall_level):
        """Get the risk factor for a rainfall level."""
        levels = self.environmental_factors.get('rainfall_levels', {})
        level_data = levels.get(rainfall_level, {})
        return level_data.get('risk_factor', 1.0)

    def get_all_crop_names(self):
        """Get all available crop names."""
        return list(self.crop_mapping.keys())

    def get_display_name(self, class_label):
        """Get the human-readable display name for a disease."""
        info = self.get_disease_info(class_label)
        if info:
            return info.get('display_name', class_label)
        # Handle healthy classes
        if self.is_healthy(class_label):
            crop = class_label.split('___')[0].replace('_', ' ').replace('(', '').replace(')', '')
            return f"{crop} - Healthy"
        return class_label
