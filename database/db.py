"""
Database module for storing diagnosis history.

Uses SQLite for zero-configuration persistence.
"""
import sqlite3
import json
import os
from datetime import datetime


class Database:
    """SQLite database manager for diagnosis records."""

    def __init__(self, db_path):
        """
        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self):
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize the database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS diagnoses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                image_filename TEXT NOT NULL,
                image_path TEXT NOT NULL,
                crop_type TEXT,
                predicted_disease TEXT NOT NULL,
                disease_display_name TEXT,
                confidence REAL NOT NULL,
                severity_percentage REAL,
                severity_level TEXT,
                risk_score REAL,
                risk_level TEXT,
                humidity REAL,
                temperature REAL,
                rainfall TEXT,
                growth_stage TEXT,
                recommendations TEXT,
                heatmap_url TEXT,
                overlay_url TEXT,
                mock_mode INTEGER DEFAULT 0,
                full_result TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def save_diagnosis(self, diagnosis_data):
        """
        Save a diagnosis record.

        Args:
            diagnosis_data: dict containing all diagnosis fields.

        Returns:
            int: The ID of the inserted record.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO diagnoses (
                timestamp, image_filename, image_path, crop_type,
                predicted_disease, disease_display_name, confidence,
                severity_percentage, severity_level,
                risk_score, risk_level,
                humidity, temperature, rainfall, growth_stage,
                recommendations, heatmap_url, overlay_url,
                mock_mode, full_result
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            diagnosis_data.get('image_filename', ''),
            diagnosis_data.get('image_path', ''),
            diagnosis_data.get('crop_type'),
            diagnosis_data.get('predicted_disease', ''),
            diagnosis_data.get('disease_display_name', ''),
            diagnosis_data.get('confidence', 0),
            diagnosis_data.get('severity_percentage'),
            diagnosis_data.get('severity_level'),
            diagnosis_data.get('risk_score'),
            diagnosis_data.get('risk_level'),
            diagnosis_data.get('humidity'),
            diagnosis_data.get('temperature'),
            diagnosis_data.get('rainfall'),
            diagnosis_data.get('growth_stage'),
            json.dumps(diagnosis_data.get('recommendations', {}), default=lambda o: o.item() if hasattr(o, 'item') else str(o)),
            diagnosis_data.get('heatmap_url'),
            diagnosis_data.get('overlay_url'),
            1 if diagnosis_data.get('mock_mode', False) else 0,
            json.dumps(diagnosis_data.get('full_result', {}), default=lambda o: o.item() if hasattr(o, 'item') else str(o)),
        ))

        diagnosis_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return diagnosis_id

    def get_all_diagnoses(self, limit=50, offset=0):
        """
        Get all diagnosis records with pagination.

        Returns:
            list: List of diagnosis records as dicts.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM diagnoses
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        ''', (limit, offset))

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_dict(row) for row in rows]

    def get_diagnosis(self, diagnosis_id):
        """
        Get a specific diagnosis by ID.

        Returns:
            dict or None: The diagnosis record.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM diagnoses WHERE id = ?', (diagnosis_id,))
        row = cursor.fetchone()
        conn.close()

        return self._row_to_dict(row) if row else None

    def delete_diagnosis(self, diagnosis_id):
        """
        Delete a diagnosis record.

        Returns:
            bool: True if a record was deleted.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('DELETE FROM diagnoses WHERE id = ?', (diagnosis_id,))
        deleted = cursor.rowcount > 0

        conn.commit()
        conn.close()

        return deleted

    def get_diagnosis_count(self):
        """Get the total number of diagnoses."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) as count FROM diagnoses')
        count = cursor.fetchone()['count']

        conn.close()
        return count

    def _row_to_dict(self, row):
        """Convert a database row to a dictionary."""
        d = dict(row)
        # Parse JSON fields
        if d.get('recommendations'):
            try:
                d['recommendations'] = json.loads(d['recommendations'])
            except (json.JSONDecodeError, TypeError):
                d['recommendations'] = {}
        if d.get('full_result'):
            try:
                d['full_result'] = json.loads(d['full_result'])
            except (json.JSONDecodeError, TypeError):
                d['full_result'] = {}
        d['mock_mode'] = bool(d.get('mock_mode', 0))
        return d
