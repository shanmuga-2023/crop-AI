"""
Risk Calculator for Crop Disease Expert System.

Combines AI prediction confidence, severity, environmental factors,
and crop growth stage into a weighted risk score with explanations.
"""
from .engine import InferenceEngine


class RiskCalculator:
    """Calculate disease risk and generate recommendations."""

    # Risk level thresholds
    RISK_LEVELS = {
        'low': {'min': 0, 'max': 25, 'color': '#10b981', 'label': 'Low Risk'},
        'moderate': {'min': 25, 'max': 50, 'color': '#f59e0b', 'label': 'Moderate Risk'},
        'high': {'min': 50, 'max': 75, 'color': '#f97316', 'label': 'High Risk'},
        'critical': {'min': 75, 'max': 100, 'color': '#ef4444', 'label': 'Critical Risk'},
    }

    # Weights for risk calculation components
    WEIGHTS = {
        'severity': 0.35,
        'confidence': 0.20,
        'humidity': 0.15,
        'temperature': 0.10,
        'rainfall': 0.10,
        'growth_stage': 0.10,
    }

    def __init__(self, knowledge_base):
        """
        Args:
            knowledge_base: KnowledgeBase instance.
        """
        self.kb = knowledge_base

    def calculate_risk(self, disease_label, severity_pct, confidence,
                       humidity=None, temperature=None, rainfall=None,
                       growth_stage=None, farming_type=None):
        """
        Calculate comprehensive risk score.

        Args:
            disease_label: Predicted disease class label.
            severity_pct: Disease severity percentage (0-100).
            confidence: Model prediction confidence (0-1).
            humidity: Relative humidity percentage (optional).
            temperature: Temperature in Celsius (optional).
            rainfall: Rainfall level string (optional).
            growth_stage: Current crop growth stage (optional).
            farming_type: Farming type e.g., 'organic_farming' (optional).

        Returns:
            dict: Risk assessment with score, level, factors, and recommendations.
        """
        # Initialize inference engine
        engine = InferenceEngine(self.kb)

        # --- Add base facts ---
        engine.add_fact('disease', disease_label, source='ai')
        engine.add_fact('severity_percentage', severity_pct, source='ai')
        engine.add_fact('confidence', confidence, source='ai')
        engine.add_fact('is_healthy', self.kb.is_healthy(disease_label), source='ai')

        # Add pathogen type
        disease_info = self.kb.get_disease_info(disease_label)
        if disease_info:
            engine.add_fact('pathogen_type', disease_info.get('pathogen_type', 'Unknown'), source='ai')

        # --- Add environmental facts ---
        if humidity is not None:
            engine.add_fact('humidity', humidity, source='user')
        else:
            engine.add_fact('humidity', 60, source='default')  # default moderate

        if temperature is not None:
            engine.add_fact('temperature', temperature, source='user')
            # Check if temperature is in favorable range for the disease
            if disease_info:
                fav = disease_info.get('favorable_conditions', {})
                in_range = fav.get('temperature_min', 0) <= temperature <= fav.get('temperature_max', 50)
                engine.add_fact('temperature_in_range', in_range, source='inferred')
        else:
            engine.add_fact('temperature', 25, source='default')
            engine.add_fact('temperature_in_range', True, source='default')

        if rainfall is not None:
            engine.add_fact('rainfall', rainfall, source='user')
        else:
            engine.add_fact('rainfall', 'moderate', source='default')

        if growth_stage is not None:
            engine.add_fact('growth_stage', growth_stage, source='user')

        # --- Run inference ---
        inference_results = engine.run()

        # --- Calculate numeric risk score ---
        risk_score = self._compute_risk_score(
            disease_label=disease_label,
            severity_pct=severity_pct,
            confidence=confidence,
            humidity=humidity or 60,
            temperature=temperature or 25,
            rainfall=rainfall or 'moderate',
            growth_stage=growth_stage,
            farming_type=farming_type,
        )

        # --- Determine risk level ---
        risk_level = self._get_risk_level(risk_score)

        # --- Get contributing factors ---
        contributing_factors = self._get_contributing_factors(
            engine, disease_label, severity_pct, humidity, temperature, rainfall, growth_stage
        )

        # --- Generate recommendations ---
        recommendations = self._generate_recommendations(
            disease_label=disease_label,
            severity_level=engine.get_fact_value('severity_level', 'mild'),
            action_urgency=engine.get_fact_value('action_urgency', 'monitor'),
            special_note=engine.get_fact_value('special_note'),
            risk_level=risk_level,
        )

        return {
            'risk_score': round(float(risk_score), 1),
            'risk_level': risk_level,
            'risk_label': self.RISK_LEVELS[risk_level]['label'],
            'risk_color': self.RISK_LEVELS[risk_level]['color'],
            'severity_level': engine.get_fact_value('severity_level', 'mild'),
            'action_urgency': engine.get_fact_value('action_urgency', 'monitor'),
            'contributing_factors': contributing_factors,
            'recommendations': recommendations,
            'inference_details': {
                'fired_rules': inference_results['fired_rules'],
                'explanations': inference_results['explanations'],
                'inferred_facts': inference_results['inferred_facts'],
            },
        }

    def _compute_risk_score(self, disease_label, severity_pct, confidence,
                            humidity, temperature, rainfall, growth_stage, farming_type):
        """Compute a numeric risk score from 0-100."""

        # If healthy, return very low risk
        if self.kb.is_healthy(disease_label):
            return max(0, severity_pct * 0.3)

        # Severity component (0-100)
        severity_score = min(severity_pct, 100)

        # Confidence component (0-100)
        confidence_score = confidence * 100

        # Humidity component (0-100)
        _, humidity_factor = self.kb.classify_humidity(humidity)
        humidity_score = humidity_factor * 60  # scale up

        # Temperature component (0-100)
        disease_info = self.kb.get_disease_info(disease_label)
        temp_score = 50  # default
        if disease_info:
            fav = disease_info.get('favorable_conditions', {})
            t_min = fav.get('temperature_min', 15)
            t_max = fav.get('temperature_max', 30)
            if t_min <= temperature <= t_max:
                temp_score = 80
            elif abs(temperature - t_min) <= 5 or abs(temperature - t_max) <= 5:
                temp_score = 60
            else:
                temp_score = 30

        # Rainfall component (0-100)
        rainfall_factor = self.kb.get_rainfall_risk_factor(rainfall)
        rainfall_score = rainfall_factor * 55

        # Growth stage component (0-100)
        stage_score = 50  # default
        if growth_stage and disease_info:
            vulnerability = disease_info.get('growth_stage_vulnerability', {})
            stage_mult = vulnerability.get(growth_stage, 0.7)
            stage_score = stage_mult * 80

        # Weighted combination
        raw_score = (
            self.WEIGHTS['severity'] * severity_score +
            self.WEIGHTS['confidence'] * confidence_score +
            self.WEIGHTS['humidity'] * humidity_score +
            self.WEIGHTS['temperature'] * temp_score +
            self.WEIGHTS['rainfall'] * rainfall_score +
            self.WEIGHTS['growth_stage'] * stage_score
        )

        # Apply farming type modifier
        if farming_type and disease_info:
            modifiers = disease_info.get('risk_modifiers', {})
            modifier = modifiers.get(farming_type, 1.0)
            raw_score *= modifier

        return min(max(raw_score, 0), 100)

    def _get_risk_level(self, risk_score):
        """Map a numeric risk score to a risk level category."""
        for level, data in self.RISK_LEVELS.items():
            if data['min'] <= risk_score < data['max']:
                return level
        return 'critical'

    def _get_contributing_factors(self, engine, disease_label, severity_pct,
                                 humidity, temperature, rainfall, growth_stage):
        """Identify the main contributing factors to the risk score."""
        factors = []

        # Severity factor
        severity_level = engine.get_fact_value('severity_level', 'mild')
        if severity_level in ['severe', 'critical']:
            factors.append({
                'factor': 'Disease Severity',
                'impact': 'high',
                'detail': f'Severity at {severity_pct:.1f}% ({severity_level}) indicates significant infection'
            })
        elif severity_level == 'moderate':
            factors.append({
                'factor': 'Disease Severity',
                'impact': 'moderate',
                'detail': f'Severity at {severity_pct:.1f}% ({severity_level}) requires attention'
            })

        # Humidity factor
        if humidity is not None and humidity >= 75:
            factors.append({
                'factor': 'High Humidity',
                'impact': 'high',
                'detail': f'Humidity at {humidity}% creates favorable conditions for disease spread'
            })
        elif humidity is not None and humidity >= 60:
            factors.append({
                'factor': 'Moderate Humidity',
                'impact': 'moderate',
                'detail': f'Humidity at {humidity}% may support pathogen activity'
            })

        # Temperature factor
        if temperature is not None:
            disease_info = self.kb.get_disease_info(disease_label)
            if disease_info:
                fav = disease_info.get('favorable_conditions', {})
                t_min = fav.get('temperature_min', 15)
                t_max = fav.get('temperature_max', 30)
                if t_min <= temperature <= t_max:
                    factors.append({
                        'factor': 'Temperature',
                        'impact': 'high',
                        'detail': f'Temperature {temperature}°C is within the optimal range ({t_min}-{t_max}°C) for this pathogen'
                    })

        # Rainfall factor
        if rainfall in ['heavy', 'moderate_to_heavy']:
            factors.append({
                'factor': 'Rainfall',
                'impact': 'high',
                'detail': f'Heavy rainfall promotes spore dispersal and leaf wetness'
            })

        # Growth stage factor
        if growth_stage:
            disease_info = self.kb.get_disease_info(disease_label)
            if disease_info:
                vulnerability = disease_info.get('growth_stage_vulnerability', {})
                stage_mult = vulnerability.get(growth_stage, 0.7)
                if stage_mult >= 0.9:
                    factors.append({
                        'factor': 'Growth Stage',
                        'impact': 'high',
                        'detail': f'The {growth_stage} stage is highly vulnerable to this disease'
                    })

        # Special: Viral diseases
        special_note = engine.get_fact_value('special_note')
        if special_note == 'no_chemical_cure':
            factors.append({
                'factor': 'Viral Disease',
                'impact': 'critical',
                'detail': 'This is a viral disease with no chemical cure. Focus on vector control and prevention.'
            })

        return factors

    def _generate_recommendations(self, disease_label, severity_level,
                                  action_urgency, special_note, risk_level):
        """Generate context-aware recommendations based on all assessed factors."""
        recommendations = {
            'urgency': action_urgency,
            'urgency_message': '',
            'immediate_actions': [],
            'treatment': {},
            'preventive_measures': [],
            'monitoring': [],
        }

        # --- Urgency message ---
        urgency_messages = {
            'immediate': '⚠️ IMMEDIATE ACTION REQUIRED — Disease is at a critical stage. Begin treatment within 24-48 hours to prevent further spread.',
            'prompt': '⏰ PROMPT ACTION RECOMMENDED — Disease has reached a level that warrants timely intervention. Initiate treatment within the next few days.',
            'monitor': '👁️ CONTINUE MONITORING — Current conditions suggest low immediate risk. Regular scouting and preventive measures are advised.',
        }
        recommendations['urgency_message'] = urgency_messages.get(
            action_urgency, urgency_messages['monitor']
        )

        # --- Immediate actions based on severity ---
        if severity_level in ['severe', 'critical']:
            recommendations['immediate_actions'] = [
                'Isolate affected plants from healthy ones if possible',
                'Remove and destroy heavily infected plant material',
                'Begin appropriate treatment immediately',
                'Increase monitoring frequency to daily inspections',
                'Document the extent of infection for records',
            ]
        elif severity_level == 'moderate':
            recommendations['immediate_actions'] = [
                'Begin preventive spraying on nearby healthy plants',
                'Remove the most affected leaves/parts',
                'Improve air circulation around plants',
                'Adjust irrigation to reduce leaf wetness',
            ]
        else:
            recommendations['immediate_actions'] = [
                'Continue regular field scouting',
                'Apply preventive measures as recommended',
                'Maintain good cultural practices',
            ]

        # --- Treatment from knowledge base ---
        if not self.kb.is_healthy(disease_label):
            recommendations['treatment'] = self.kb.get_treatment(disease_label)

        # --- Preventive measures ---
        recommendations['preventive_measures'] = self.kb.get_preventive_measures(disease_label)

        # --- Monitoring recommendations ---
        if risk_level == 'critical':
            recommendations['monitoring'] = [
                'Inspect all plants daily for disease progression',
                'Monitor weather conditions closely',
                'Prepare contingency plans for crop loss',
                'Consider consulting a local agricultural extension officer',
            ]
        elif risk_level == 'high':
            recommendations['monitoring'] = [
                'Inspect plants every 2-3 days',
                'Monitor humidity and temperature trends',
                'Track disease spread to neighboring plants',
                'Evaluate treatment effectiveness after 5-7 days',
            ]
        elif risk_level == 'moderate':
            recommendations['monitoring'] = [
                'Weekly plant inspections recommended',
                'Watch for weather conditions that may worsen disease',
                'Monitor effectiveness of preventive measures',
            ]
        else:
            recommendations['monitoring'] = [
                'Routine bi-weekly inspections sufficient',
                'Maintain general crop health monitoring',
            ]

        # --- Special notes for viral diseases ---
        if special_note == 'no_chemical_cure':
            recommendations['special_notes'] = [
                'This disease is caused by a virus — there is no direct chemical cure.',
                'Focus on controlling the insect vector (e.g., whiteflies, aphids).',
                'Remove and destroy infected plants to prevent spread.',
                'Use virus-resistant varieties for future plantings.',
            ]

        return recommendations

    def what_if_analysis(self, disease_label, severity_pct, confidence,
                         original_conditions, modified_conditions):
        """
        Perform what-if analysis comparing original vs modified conditions.

        Args:
            disease_label: Disease class label.
            severity_pct: Severity percentage.
            confidence: Prediction confidence.
            original_conditions: dict with humidity, temperature, rainfall, growth_stage.
            modified_conditions: dict with modified values.

        Returns:
            dict: Comparison of original vs modified risk assessments.
        """
        # Calculate original risk
        original_risk = self.calculate_risk(
            disease_label=disease_label,
            severity_pct=original_conditions.get('severity_pct', severity_pct),
            confidence=confidence,
            humidity=original_conditions.get('humidity'),
            temperature=original_conditions.get('temperature'),
            rainfall=original_conditions.get('rainfall'),
            growth_stage=original_conditions.get('growth_stage'),
        )

        # Calculate modified risk
        modified_risk = self.calculate_risk(
            disease_label=disease_label,
            severity_pct=modified_conditions.get('severity_pct', severity_pct),
            confidence=confidence,
            humidity=modified_conditions.get('humidity'),
            temperature=modified_conditions.get('temperature'),
            rainfall=modified_conditions.get('rainfall'),
            growth_stage=modified_conditions.get('growth_stage'),
        )

        # Compare
        score_change = modified_risk['risk_score'] - original_risk['risk_score']
        risk_changed = original_risk['risk_level'] != modified_risk['risk_level']

        return {
            'original': original_risk,
            'modified': modified_risk,
            'score_change': round(score_change, 1),
            'risk_level_changed': risk_changed,
            'direction': 'increased' if score_change > 0 else ('decreased' if score_change < 0 else 'unchanged'),
            'summary': self._generate_whatif_summary(original_risk, modified_risk, score_change),
        }

    def _generate_whatif_summary(self, original, modified, score_change):
        """Generate a human-readable summary of what-if changes."""
        if abs(score_change) < 2:
            return "The modified conditions have minimal impact on the overall risk assessment."
        elif score_change > 0:
            return (
                f"Risk has INCREASED from {original['risk_score']}% ({original['risk_label']}) "
                f"to {modified['risk_score']}% ({modified['risk_label']}). "
                f"The modified conditions are more favorable for disease development."
            )
        else:
            return (
                f"Risk has DECREASED from {original['risk_score']}% ({original['risk_label']}) "
                f"to {modified['risk_score']}% ({modified['risk_label']}). "
                f"The modified conditions are less favorable for disease development."
            )
