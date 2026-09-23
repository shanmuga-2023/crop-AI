"""
Diagnosis API route.

Handles image upload, disease classification, Grad-CAM generation,
severity estimation, and expert system consultation.
"""
import os
import uuid
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from config import Config

diagnosis_bp = Blueprint('diagnosis', __name__)


def allowed_file(filename):
    """Check if a filename has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


@diagnosis_bp.route('/api/diagnose', methods=['POST'])
def diagnose():
    """
    Full diagnosis pipeline endpoint.

    Accepts:
        - image: The leaf image file (required)
        - crop_type: Name of the crop (optional)
        - humidity: Relative humidity % (optional)
        - temperature: Temperature in °C (optional)
        - rainfall: Rainfall level string (optional)
        - growth_stage: Current growth stage (optional)

    Returns:
        JSON with prediction, severity, Grad-CAM, risk assessment, and recommendations.
    """
    # --- Validate image upload ---
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({
            'error': f'File type not allowed. Accepted: {", ".join(Config.ALLOWED_EXTENSIONS)}'
        }), 400

    # --- Save uploaded file ---
    ext = file.filename.rsplit('.', 1)[1].lower()
    unique_filename = f"{uuid.uuid4().hex[:12]}.{ext}"
    safe_filename = secure_filename(unique_filename)
    image_path = os.path.join(Config.UPLOAD_FOLDER, safe_filename)
    file.save(image_path)

    # --- Parse optional metadata ---
    crop_type = request.form.get('crop_type', '').strip() or None
    humidity = _parse_float(request.form.get('humidity'))
    temperature = _parse_float(request.form.get('temperature'))
    rainfall = request.form.get('rainfall', '').strip() or None
    growth_stage = request.form.get('growth_stage', '').strip() or None

    try:
        # --- Step 1: Disease Classification ---
        prediction = current_app.classifier.predict(image_path, top_k=3)

        top_disease = prediction['top_prediction']['class_label']
        confidence = prediction['top_prediction']['confidence']

        # Get display name from knowledge base
        disease_display = current_app.knowledge_base.get_display_name(top_disease)
        is_healthy = current_app.knowledge_base.is_healthy(top_disease)

        # --- Step 2: Grad-CAM Explainability ---
        gradcam_result = current_app.gradcam.generate(
            image_path=image_path,
            model=current_app.classifier.model,
            class_index=prediction['top_prediction']['class_index'],
            prediction_result=prediction,
        )
        # Remove raw array from response (too large for JSON)
        gradcam_response = {k: v for k, v in gradcam_result.items() if k != 'heatmap_array'}

        # --- Step 3: Severity Estimation ---
        severity = current_app.severity_estimator.estimate(image_path)

        # --- Step 4: Expert System Risk Assessment ---
        risk_assessment = current_app.risk_calculator.calculate_risk(
            disease_label=top_disease,
            severity_pct=severity['severity_percentage'],
            confidence=confidence,
            humidity=humidity,
            temperature=temperature,
            rainfall=rainfall,
            growth_stage=growth_stage,
        )

        # --- Step 5: Get disease details from knowledge base ---
        disease_info = current_app.knowledge_base.get_disease_info(top_disease)
        disease_details = None
        if disease_info:
            disease_details = {
                'display_name': disease_info.get('display_name'),
                'crop': disease_info.get('crop'),
                'scientific_name': disease_info.get('scientific_name'),
                'pathogen_type': disease_info.get('pathogen_type'),
                'symptoms': disease_info.get('symptoms', []),
                'causes': disease_info.get('causes', []),
                'favorable_conditions': disease_info.get('favorable_conditions', {}),
            }

        # --- Compile full response ---
        result = {
            'success': True,
            'image_url': f'/uploads/{safe_filename}',
            'prediction': {
                'disease_label': top_disease,
                'disease_name': disease_display,
                'is_healthy': is_healthy,
                'confidence': round(confidence, 4),
                'confidence_pct': round(confidence * 100, 1),
                'top_predictions': [
                    {
                        'label': p['class_label'],
                        'name': current_app.knowledge_base.get_display_name(p['class_label']),
                        'confidence': round(p['confidence'], 4),
                        'confidence_pct': round(p['confidence'] * 100, 1),
                    }
                    for p in prediction['top_k_predictions']
                ],
                'mock_mode': prediction['mock_mode'],
            },
            'gradcam': gradcam_response,
            'severity': severity,
            'risk_assessment': risk_assessment,
            'disease_details': disease_details,
            'input_conditions': {
                'crop_type': crop_type,
                'humidity': humidity,
                'temperature': temperature,
                'rainfall': rainfall,
                'growth_stage': growth_stage,
            },
        }

        # --- Save to database ---
        db_data = {
            'image_filename': safe_filename,
            'image_path': image_path,
            'crop_type': crop_type,
            'predicted_disease': top_disease,
            'disease_display_name': disease_display,
            'confidence': confidence,
            'severity_percentage': severity['severity_percentage'],
            'severity_level': severity['severity_level'],
            'risk_score': risk_assessment['risk_score'],
            'risk_level': risk_assessment['risk_level'],
            'humidity': humidity,
            'temperature': temperature,
            'rainfall': rainfall,
            'growth_stage': growth_stage,
            'recommendations': risk_assessment.get('recommendations', {}),
            'heatmap_url': gradcam_response.get('heatmap_url'),
            'overlay_url': gradcam_response.get('overlay_url'),
            'mock_mode': prediction['mock_mode'],
            'full_result': result,
        }

        diagnosis_id = current_app.db.save_diagnosis(db_data)
        result['diagnosis_id'] = diagnosis_id

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Diagnosis failed: {str(e)}',
        }), 500


def _parse_float(value):
    """Safely parse a float value from form data."""
    if value is None or value == '':
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
