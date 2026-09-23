"""
History API route.

Provides endpoints for retrieving and managing past diagnosis records.
"""
from flask import Blueprint, request, jsonify, current_app

history_bp = Blueprint('history', __name__)


@history_bp.route('/api/history', methods=['GET'])
def get_history():
    """
    Get diagnosis history with pagination.

    Query params:
        - limit: Number of records (default 20, max 100)
        - offset: Pagination offset (default 0)
    """
    try:
        limit = min(int(request.args.get('limit', 20)), 100)
        offset = max(int(request.args.get('offset', 0)), 0)
    except (ValueError, TypeError):
        limit, offset = 20, 0

    diagnoses = current_app.db.get_all_diagnoses(limit=limit, offset=offset)
    total = current_app.db.get_diagnosis_count()

    return jsonify({
        'success': True,
        'diagnoses': diagnoses,
        'total': total,
        'limit': limit,
        'offset': offset,
        'has_more': (offset + limit) < total,
    })


@history_bp.route('/api/history/<int:diagnosis_id>', methods=['GET'])
def get_diagnosis_detail(diagnosis_id):
    """Get a specific diagnosis by ID."""
    diagnosis = current_app.db.get_diagnosis(diagnosis_id)

    if not diagnosis:
        return jsonify({'error': 'Diagnosis not found'}), 404

    return jsonify({
        'success': True,
        'diagnosis': diagnosis,
    })


@history_bp.route('/api/history/<int:diagnosis_id>', methods=['DELETE'])
def delete_diagnosis(diagnosis_id):
    """Delete a diagnosis record."""
    deleted = current_app.db.delete_diagnosis(diagnosis_id)

    if not deleted:
        return jsonify({'error': 'Diagnosis not found'}), 404

    return jsonify({
        'success': True,
        'message': f'Diagnosis {diagnosis_id} deleted',
    })
