# 🌿 CropAI Expert — Explainable AI-Based Crop Disease Diagnosis System

An intelligent decision-support platform that integrates **deep learning**, **Grad-CAM explainability**, **severity assessment**, and a **rule-based expert system** into a single web application for crop disease diagnosis and management.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-000000?logo=flask&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Usage Guide](#usage-guide)
- [API Endpoints](#api-endpoints)
- [Knowledge Base](#knowledge-base)
- [Model Training](#model-training)
- [Screenshots](#screenshots)

---

## Overview

Unlike conventional crop disease classifiers that only provide a disease label, CropAI Expert functions as a **complete intelligent decision-support platform**, providing:

1. **Disease Identification** — Deep learning classification with confidence scores
2. **Severity Estimation** — Computer vision analysis of infected leaf area
3. **Explainable Evidence** — Grad-CAM heatmaps showing why the AI made its decision
4. **Risk Assessment** — Expert system combining AI predictions with environmental data
5. **Actionable Recommendations** — Context-aware treatment plans and preventive measures
6. **What-If Analysis** — Interactive tool to explore how changing conditions affects risk

---

## Features

### 🔬 AI Disease Detection
- **ResNet50** transfer learning model trained on the PlantVillage dataset
- Supports **38 disease classes** across **14 crop species**
- Top-3 predictions with confidence scores
- Mock mode for demonstration when model weights are unavailable

### 🧠 Explainable AI (Grad-CAM)
- Gradient-weighted Class Activation Mapping
- Visual heatmap overlay on original leaf image
- Adjustable opacity slider for detailed inspection
- Highlights disease-relevant regions (lesions, spots, discoloration)

### 📊 Severity Assessment
- HSV color space segmentation using NumPy/PIL
- Separates healthy leaf tissue from diseased areas
- Quantified severity percentage with animated gauge
- Five severity levels: Healthy → Mild → Moderate → Severe → Critical

### 🛡️ Expert System
- **Forward-chaining inference engine** with prioritized rules
- JSON-based knowledge base covering 26+ diseases
- Combines AI prediction with environmental factors:
  - Humidity, temperature, rainfall, crop growth stage
- Weighted risk scoring with transparent contributing factors
- Context-aware treatment recommendations (organic + chemical)

### 🔮 What-If Analysis
- Interactive sliders to modify environmental conditions
- Real-time risk recalculation via API
- Side-by-side comparison of original vs. modified risk
- Helps farmers understand which factors matter most

### 🎨 Premium UI
- Dark glassmorphism theme with gradient accents
- Animated landing page with particle effects and typewriter text
- Drag-and-drop image upload with live preview
- Responsive design (mobile-friendly)
- Smooth micro-animations and transitions

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Frontend (HTML/CSS/JS)          │
│  Landing Page │ Diagnosis Dashboard │ History    │
└──────────────────────┬──────────────────────────┘
                       │ HTTP API
┌──────────────────────┴──────────────────────────┐
│                 Flask Backend                    │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Classifier│  │ Grad-CAM │  │   Severity   │  │
│  │ (ResNet50)│  │ Generator│  │  Estimator   │  │
│  └─────┬─────┘  └─────┬────┘  └──────┬───────┘  │
│        │              │              │           │
│  ┌─────┴──────────────┴──────────────┴───────┐  │
│  │         Expert System Engine               │  │
│  │  ┌──────────┐  ┌─────────────────────┐    │  │
│  │  │ Inference │  │   Knowledge Base    │    │  │
│  │  │  Engine   │  │   (diseases.json)   │    │  │
│  │  └──────────┘  └─────────────────────┘    │  │
│  │  ┌──────────────────────────────────┐     │  │
│  │  │     Risk Calculator              │     │  │
│  │  │  + What-If Analysis Engine       │     │  │
│  │  └──────────────────────────────────┘     │  │
│  └───────────────────────────────────────────┘  │
│                                                  │
│  ┌───────────────────────────────────────────┐  │
│  │            SQLite Database                │  │
│  │         (Diagnosis History)               │  │
│  └───────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component           | Technology                              |
|--------------------|-----------------------------------------|
| **Language**        | Python 3.8+                             |
| **Web Framework**   | Flask 3.1                               |
| **Deep Learning**   | TensorFlow / Keras (ResNet50)           |
| **Explainability**  | Grad-CAM (GradientTape)                 |
| **Image Analysis**  | NumPy, Pillow (PIL)                     |
| **Expert System**   | Custom forward-chaining engine (Python) |
| **Database**        | SQLite                                  |
| **Frontend**        | Vanilla HTML5, CSS3, JavaScript (ES6+)  |
| **Typography**      | Google Fonts (Inter, JetBrains Mono)    |

---

## Project Structure

```
it agr/
├── app.py                          # Flask application entry point
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── models/                         # Deep Learning models
│   ├── classifier.py               # Disease classification (ResNet50)
│   ├── gradcam.py                  # Grad-CAM explainability
│   ├── severity.py                 # Severity estimation
│   └── weights/                    # Model weights (auto-created)
│
├── expert_system/                  # Rule-based Expert System
│   ├── engine.py                   # Forward-chaining inference engine
│   ├── knowledge_base.py           # Knowledge base loader
│   ├── risk_calculator.py          # Risk assessment & what-if
│   └── knowledge/
│       └── diseases.json           # Disease knowledge base (26+ diseases)
│
├── routes/                         # Flask API routes
│   ├── diagnosis.py                # POST /api/diagnose
│   ├── whatif.py                   # POST /api/what-if
│   └── history.py                  # GET/DELETE /api/history
│
├── database/
│   └── db.py                       # SQLite database layer
│
├── static/
│   ├── css/style.css               # Complete design system
│   ├── js/
│   │   ├── app.js                  # Diagnosis page logic
│   │   └── landing.js              # Landing page animations
│   ├── images/                     # Static assets
│   └── gradcam_outputs/            # Generated heatmaps
│
├── templates/
│   ├── base.html                   # Base layout
│   ├── landing.html                # Landing page
│   ├── index.html                  # Diagnosis page
│   └── history.html                # History page
│
└── uploads/                        # User-uploaded images
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. **Clone or download the project:**
   ```bash
   cd "it agr"
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # macOS/Linux
   # venv\Scripts\activate    # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** TensorFlow is listed in requirements but is only needed if you have trained model weights. The application runs in **mock mode** without TensorFlow installed.

4. **For minimal installation (no TensorFlow):**
   ```bash
   pip install Flask flask-cors Pillow numpy
   ```

---

## Running the Application

```bash
python app.py
```

The application will start at **http://127.0.0.1:5001**

### Pages

| URL          | Description                        |
|--------------|------------------------------------|
| `/`          | Landing page                       |
| `/diagnose`  | Main diagnosis tool                |
| `/history`   | Past diagnosis records             |

### Mock Mode

If no trained model weights are found at `models/weights/resnet50_plantvillage.h5`, the system automatically runs in **mock mode**:

- Disease predictions are generated based on image color analysis
- Grad-CAM heatmaps highlight non-green areas (simulating disease detection)
- All other features (severity, expert system, what-if) work fully
- A "DEMO MODE" badge is shown in the UI

---

## Usage Guide

### 1. Upload a Leaf Image
- Navigate to `/diagnose`
- Drag-and-drop or click to upload a leaf photo (JPG, PNG, WebP)
- Optionally provide environmental data (humidity, temperature, rainfall, growth stage)

### 2. View Diagnosis Results
After clicking "Analyze Leaf", the system displays:
- **Prediction Card** — Disease name, confidence score, top-3 predictions
- **Grad-CAM Card** — Original image alongside AI heatmap overlay
- **Severity Gauge** — Animated circular gauge showing % of leaf affected
- **Risk Assessment** — Risk score (0-100) with contributing factors
- **Recommendations** — Expandable accordion with immediate actions, treatments, preventive measures
- **Disease Details** — Symptoms, causes, favorable conditions

### 3. What-If Analysis
- Adjust humidity, temperature, severity, and rainfall sliders
- See real-time risk score changes compared to the original assessment
- Understand which environmental factors most affect disease risk

---

## API Endpoints

### `POST /api/diagnose`
Upload an image for full diagnosis.

**Form Data:**
| Field          | Type   | Required | Description                    |
|----------------|--------|----------|--------------------------------|
| `image`        | File   | Yes      | Leaf image (JPG/PNG/WebP)      |
| `crop_type`    | String | No       | Crop name (e.g., "Tomato")     |
| `humidity`     | Float  | No       | Relative humidity (0-100)      |
| `temperature`  | Float  | No       | Temperature in °C              |
| `rainfall`     | String | No       | dry/light/moderate/heavy       |
| `growth_stage` | String | No       | seedling/vegetative/flowering/fruiting/mature |

**Response:** JSON with prediction, Grad-CAM URLs, severity, risk assessment, recommendations.

---

### `POST /api/what-if`
Compare risk under different conditions.

**JSON Body:**
```json
{
  "disease_label": "Tomato___Early_blight",
  "severity_pct": 25.5,
  "confidence": 0.87,
  "original": { "humidity": 65, "temperature": 25, "rainfall": "moderate" },
  "modified": { "humidity": 90, "temperature": 30, "rainfall": "heavy" }
}
```

---

### `GET /api/history?limit=20&offset=0`
Retrieve past diagnoses with pagination.

### `GET /api/history/<id>`
Get a specific diagnosis record.

### `DELETE /api/history/<id>`
Delete a diagnosis record.

### `GET /api/model-info`
Get model metadata (type, classes, mock mode status).

### `GET /api/crops`
Get available crop names and growth stages.

---

## Knowledge Base

The expert system knowledge base (`expert_system/knowledge/diseases.json`) contains detailed information for **26 diseases** across **14 crops**:

### Supported Crops & Diseases

| Crop       | Diseases                                                       |
|------------|---------------------------------------------------------------|
| Apple      | Apple Scab, Black Rot, Cedar Apple Rust                       |
| Cherry     | Powdery Mildew                                                |
| Corn       | Gray Leaf Spot, Common Rust, Northern Leaf Blight             |
| Grape      | Black Rot, Esca (Black Measles), Leaf Blight                  |
| Orange     | Citrus Greening (HLB)                                         |
| Peach      | Bacterial Spot                                                |
| Pepper     | Bacterial Spot                                                |
| Potato     | Early Blight, Late Blight                                     |
| Squash     | Powdery Mildew                                                |
| Strawberry | Leaf Scorch                                                   |
| Tomato     | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, TYLCV, Mosaic Virus |

Each disease entry includes:
- Scientific name and pathogen type
- Symptom descriptions
- Causes and transmission methods
- Favorable environmental conditions
- Growth stage vulnerability multipliers
- Severity classification thresholds
- Preventive measures
- Treatment recommendations (organic + chemical)
- Risk modifiers (farming type, irrigation)

---

## Model Training

To train the ResNet50 model on the PlantVillage dataset:

1. **Install training dependencies:**
   ```bash
   pip install tensorflow matplotlib scikit-learn tensorflow-datasets
   ```

2. **Download the dataset:**
   The PlantVillage dataset is available on:
   - [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
   - [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/plant_village)

3. **Train the model:**
   ```bash
   python training/train_model.py
   ```

4. **Model weights** will be saved to `models/weights/resnet50_plantvillage.h5`

> **Note:** Training requires GPU access (Google Colab recommended). The application works fully in mock mode without trained weights.

---

## Screenshots

### Landing Page
- Premium dark theme with animated particle background
- Typewriter hero headline with gradient text
- Feature cards, how-it-works timeline, stats counter
- Interactive what-if analysis preview

### Diagnosis Dashboard
- Drag-and-drop image upload with live preview
- Disease prediction with confidence bars
- Grad-CAM heatmap viewer with opacity control
- Animated severity gauge
- Risk assessment with contributing factors
- Expandable treatment recommendations
- Interactive what-if analysis tool

### History Page
- Card grid of past diagnoses
- Quick-view metrics (confidence, severity, risk)
- Pagination support

---

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## License

This project is developed for academic and research purposes. See the project documentation for details.

---

<p align="center">
  Built with 🌿 by CropAI Expert System
</p>
