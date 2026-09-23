"""
What-If Analysis API route.

Allows users to modify environmental conditions and observe
how risk assessment and recommendations change.
"""
from flask import Blueprint, request, jsonify, current_app

whatif_bp = Blueprint('whatif', __name__)


@whatif_bp.route('/api/what-if', methods=['POST'])
def what_if_analysis():
    """
    What-If analysis endpoint.

    Accepts JSON with:
        - disease_label: The diagnosed disease
        - severity_pct: Severity percentage
        - confidence: Prediction confidence
        - original: dict with original conditions
        - modified: dict with modified conditions

    Returns:
        JSON comparing original vs modified risk assessments.
    """
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400

    required_fields = ['disease_label', 'severity_pct', 'confidence']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400

    try:
        result = current_app.risk_calculator.what_if_analysis(
            disease_label=data['disease_label'],
            severity_pct=float(data['severity_pct']),
            confidence=float(data['confidence']),
            original_conditions=data.get('original', {}),
            modified_conditions=data.get('modified', {}),
        )

        return jsonify({
            'success': True,
            'analysis': result,
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'What-if analysis failed: {str(e)}',
        }), 500


@whatif_bp.route('/api/risk-calculate', methods=['POST'])
def calculate_risk():
    """
    Standalone risk calculation endpoint.

    Useful for recalculating risk with specific parameters
    without re-running the full diagnosis pipeline.
    """
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400

    if 'disease_label' not in data:
        return jsonify({'error': 'Missing disease_label'}), 400

    try:
        risk = current_app.risk_calculator.calculate_risk(
            disease_label=data['disease_label'],
            severity_pct=float(data.get('severity_pct', 20)),
            confidence=float(data.get('confidence', 0.8)),
            humidity=data.get('humidity'),
            temperature=data.get('temperature'),
            rainfall=data.get('rainfall'),
            growth_stage=data.get('growth_stage'),
            farming_type=data.get('farming_type'),
        )

        return jsonify({
            'success': True,
            'risk_assessment': risk,
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Risk calculation failed: {str(e)}',
        }), 500
