"""
Model Training Script for Crop Disease Diagnosis.

Trains a Transfer Learning model (ResNet50 or MobileNetV2) on the PlantVillage dataset
and saves the trained weights and class index mapping for the CropAI web application.

Usage:
    # Train using standard ResNet50:
    python training/train_model.py

    # Quick test run (fast training on laptop CPU with subset of images):
    python training/train_model.py --quick --epochs 3

    # Train with MobileNetV2 (recommended for faster training on Apple Silicon / CPU):
    python training/train_model.py --arch mobilenetv2 --epochs 5
"""
import os
import sys
import json
import shutil
import argparse
from pathlib import Path
import numpy as np

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Canonical class label mapping to match expert_system/knowledge/diseases.json
CANONICAL_MAPPING = {
    'Pepper__bell___Bacterial_spot': 'Pepper,_bell___Bacterial_spot',
    'Pepper__bell___healthy': 'Pepper,_bell___healthy',
    'Potato___Early_blight': 'Potato___Early_blight',
    'Potato___Late_blight': 'Potato___Late_blight',
    'Potato___healthy': 'Potato___healthy',
    'Tomato_Bacterial_spot': 'Tomato___Bacterial_spot',
    'Tomato_Early_blight': 'Tomato___Early_blight',
    'Tomato_Late_blight': 'Tomato___Late_blight',
    'Tomato_Leaf_Mold': 'Tomato___Leaf_Mold',
    'Tomato_Septoria_leaf_spot': 'Tomato___Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Tomato___Spider_mites_Two-spotted_spider_mite',
    'Tomato__Target_Spot': 'Tomato___Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato__Tomato_mosaic_virus': 'Tomato___Tomato_mosaic_virus',
    'Tomato_healthy': 'Tomato___healthy',
}


def find_dataset_dir(base_dir):
    """Locate the PlantVillage dataset directory."""
    candidates = [
        os.path.join(base_dir, 'PlantVillage'),
        os.path.join(base_dir, 'plantvillage'),
        os.path.join(base_dir, 'dataset'),
        os.path.join(base_dir, 'data', 'PlantVillage'),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def get_class_folders(dataset_dir):
    """
    Find valid class subdirectories, ignoring nested duplicate PlantVillage folders.
    """
    valid_folders = []
    for item in sorted(os.listdir(dataset_dir)):
        item_path = os.path.join(dataset_dir, item)
        if os.path.isdir(item_path) and item.lower() != 'plantvillage' and not item.startswith('.'):
            # Check that it contains image files
            images = [f for f in os.listdir(item_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if len(images) > 0:
                valid_folders.append(item)
    return valid_folders


def prepare_data_generators(dataset_dir, class_folders, img_size=(224, 224), batch_size=32, quick=False, max_per_class=100):
    """
    Prepare tf.data.Dataset for training and validation using image_dataset_from_directory.
    """
    import tensorflow as tf

    # If quick mode is requested, create a temporary staging directory with sample images
    if quick:
        print(f"[INFO] Quick mode enabled: using up to {max_per_class} images per class for fast training.")
        staging_dir = os.path.join(os.path.dirname(dataset_dir), 'training', '.staging_quick')
        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir)
        os.makedirs(staging_dir, exist_ok=True)

        for folder in class_folders:
            src_folder = os.path.join(dataset_dir, folder)
            dst_folder = os.path.join(staging_dir, folder)
            os.makedirs(dst_folder, exist_ok=True)
            files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))][:max_per_class]
            for f in files:
                shutil.copy(os.path.join(src_folder, f), os.path.join(dst_folder, f))
        train_source_dir = staging_dir
    else:
        train_source_dir = dataset_dir

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_source_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=class_folders,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset='training'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_source_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=class_folders,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False,
        seed=42,
        validation_split=0.2,
        subset='validation'
    )

    # Preprocessing: Rescale [0, 255] -> [0, 1] and ImageNet mean/std
    rescaling = tf.keras.layers.Rescaling(1.0 / 255.0)

    # Data augmentation for training
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    train_ds = train_ds.map(lambda x, y: (data_augmentation(rescaling(x), training=True), y),
                            num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (rescaling(x), y),
                        num_parallel_calls=tf.data.AUTOTUNE)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds



def build_model(arch='resnet50', num_classes=15, img_size=(224, 224)):
    """Build transfer learning model."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    input_shape = (img_size[0], img_size[1], 3)

    if arch.lower() == 'mobilenetv2':
        print("[INFO] Building MobileNetV2 architecture...")
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    else:
        print("[INFO] Building ResNet50 architecture...")
        base_model = tf.keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )

    # Freeze base model initially
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    # ImageNet normalization:
    # x = layers.Rescaling(1./255)(inputs)  # handled by datagen
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model


def main():
    parser = argparse.ArgumentParser(description='Train PlantVillage Crop Disease Classifier')
    parser.add_argument('--dataset', type=str, default=None, help='Path to PlantVillage dataset folder')
    parser.add_argument('--arch', type=str, default='resnet50', choices=['resnet50', 'mobilenetv2'], help='Base CNN architecture')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--quick', action='store_true', help='Fast mode using small sample per class (for laptop testing)')
    parser.add_argument('--output', type=str, default=None, help='Output path for weights .h5 file')
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # 1. Dataset location
    dataset_dir = args.dataset or find_dataset_dir(base_dir)
    if not dataset_dir or not os.path.exists(dataset_dir):
        print(f"[ERROR] Could not find PlantVillage dataset at {dataset_dir}")
        print("Please place the dataset folder inside the project root as 'PlantVillage' or pass --dataset <path>")
        sys.exit(1)

    print(f"[INFO] Using dataset directory: {dataset_dir}")

    # 2. Get class folders
    raw_class_folders = get_class_folders(dataset_dir)
    if not raw_class_folders:
        print(f"[ERROR] No valid class folders found inside {dataset_dir}")
        sys.exit(1)

    print(f"[INFO] Found {len(raw_class_folders)} classes:")
    for f in raw_class_folders:
        canonical = CANONICAL_MAPPING.get(f, f)
        print(f"  • {f} -> {canonical}")

    # Canonical ordered labels list corresponding to generator indices
    ordered_canonical_labels = [CANONICAL_MAPPING.get(f, f) for f in raw_class_folders]

    # 3. Output paths
    weights_dir = os.path.join(base_dir, 'models', 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    output_weights_path = args.output or os.path.join(weights_dir, 'resnet50_plantvillage.h5')
    classes_json_path = os.path.join(weights_dir, 'classes.json')

    # 4. Prepare data
    train_gen, val_gen = prepare_data_generators(
        dataset_dir,
        raw_class_folders,
        img_size=(224, 224),
        batch_size=args.batch_size,
        quick=args.quick
    )

    # 5. Build model
    model, base_model = build_model(
        arch=args.arch,
        num_classes=len(raw_class_folders),
        img_size=(224, 224)
    )

    # 6. Train model
    print(f"\n[INFO] Starting training ({args.epochs} epochs, batch size {args.batch_size})...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        verbose=1
    )

    # 7. Save trained weights & class labels
    print(f"\n[INFO] Saving model to: {output_weights_path}")
    model.save(output_weights_path)

    print(f"[INFO] Saving class labels to: {classes_json_path}")
    with open(classes_json_path, 'w') as f:
        json.dump(ordered_canonical_labels, f, indent=2)

    # Clean up quick staging directory if created
    staging_dir = os.path.join(base_dir, 'training', '.staging_quick')
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)

    print("\n✅ Training complete!")
    print(f"   Model Weights: {output_weights_path}")
    print(f"   Class Labels:  {classes_json_path}")
    print("\nYou can now start the web application:")
    print("   python app.py\n")


if __name__ == '__main__':
    main()
