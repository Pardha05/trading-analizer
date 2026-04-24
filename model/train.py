"""
Model Training Script (Detection Version)
Trains two MobileNetV2-based models with dual heads:
  1. Classification Head (Pattern type)
  2. Localization Head (Bounding box coordinates)
"""
import os
import sys
import json
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ─── Configuration ───────────────────────────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'saved_models')


def build_model(num_classes):
    """Build a MobileNetV2-based model with classification and detection heads."""
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Freeze base layers for transfer learning initial phase
    base_model.trainable = False

    # Input layer
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    
    # Common dense feature layer
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    
    # 1. Classification Head
    class_head = Dense(128, activation='relu')(x)
    class_head = Dropout(0.3)(class_head)
    class_out = Dense(num_classes, activation='softmax', name='class_out')(class_head)
    
    # 2. Bounding Box Regression Head
    bbox_head = Dense(128, activation='relu')(x)
    bbox_head = Dropout(0.3)(bbox_head)
    bbox_out = Dense(4, activation='sigmoid', name='bbox_out')(bbox_head)
    
    model = Model(inputs=inputs, outputs=[class_out, bbox_out])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss={
            'class_out': 'categorical_crossentropy',
            'bbox_out': 'mse'
        },
        metrics={
            'class_out': 'accuracy',
            'bbox_out': 'mae'
        },
        loss_weights={
            'class_out': 1.0,
            'bbox_out': 5.0  # Encourage model to focus on box accuracy
        }
    )

    return model


def load_dataset_samples(data_path):
    """Walk directory to find all image/json pairs and labels."""
    image_paths = []
    json_paths = []
    labels = []
    
    classes = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    for cls in classes:
        cls_dir = os.path.join(data_path, cls)
        for f in os.listdir(cls_dir):
            if f.endswith('.png'):
                img_p = os.path.join(cls_dir, f)
                json_p = img_p.replace('.png', '.json')
                if os.path.exists(json_p):
                    image_paths.append(img_p)
                    json_paths.append(json_p)
                    labels.append(class_to_idx[cls])
                    
    return np.array(image_paths), np.array(json_paths), np.array(labels), classes


def parse_function(img_path, json_path, label, num_classes):
    """TF Dataset parser function."""
    # Load and decode image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 255.0
    
    # Load and parse JSON for bbox
    json_raw = tf.io.read_file(json_path)
    # Simple extraction of coordinates from JSON string using regex/split in py_function
    # because tf.io.decode_json_query can be flaky on different TF versions
    def get_bbox(json_str):
        data = json.loads(json_str.numpy().decode('utf-8'))
        return np.array(data['bbox'], dtype=np.float32)
    
    bbox = tf.py_function(get_bbox, [json_raw], tf.float32)
    bbox.set_shape([4])
    
    # One-hot encode label
    label_one_hot = tf.one_hot(label, num_classes)
    
    return img, {'class_out': label_one_hot, 'bbox_out': bbox}


def create_tf_dataset(image_paths, json_paths, labels, num_classes, is_training=True):
    """Create a high-performance tf.data pipeline."""
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, json_paths, labels))
    
    if is_training:
        dataset = dataset.shuffle(len(image_paths))
        
    dataset = dataset.map(
        lambda i, j, l: parse_function(i, j, l, num_classes),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset


def train_model(model_name, data_path, save_path):
    """Unified training function for multi-head model."""
    print(f"\n{'=' * 60}")
    print(f"  Training (Detection): {model_name}")
    print(f"  Data: {data_path}")
    print(f"{'=' * 60}")

    img_paths, json_paths, labels, classes = load_dataset_samples(data_path)
    num_classes = len(classes)
    
    # Split data manually
    indices = np.arange(len(img_paths))
    np.random.shuffle(indices)
    
    split = int(0.8 * len(img_paths))
    train_idx, val_idx = indices[:split], indices[split:]
    
    train_ds = create_tf_dataset(img_paths[train_idx], json_paths[train_idx], labels[train_idx], num_classes)
    val_ds = create_tf_dataset(img_paths[val_idx], json_paths[val_idx], labels[val_idx], num_classes, is_training=False)

    print(f"\n  Classes ({num_classes}):")
    for idx, cls in enumerate(classes):
        print(f"    {idx}: {cls}")
    print(f"\n  Training samples: {len(train_idx)}")
    print(f"  Validation samples: {len(val_idx)}")

    model = build_model(num_classes)

    checkpoint_file = os.path.join(save_path, f'{model_name}_best.keras')
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_file,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    history = model.fit(
        train_ds,
        epochs=5,  # Reduced for faster verification
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )

    # Save model
    os.makedirs(save_path, exist_ok=True)
    model_file = os.path.join(save_path, f'{model_name}.keras')
    model.save(model_file)
    print(f"\n  Model saved to: {model_file}")

    # Save class labels
    labels_file = os.path.join(save_path, f'{model_name}_labels.json')
    class_labels = {idx: name for idx, name in enumerate(classes)}
    with open(labels_file, 'w') as f:
        json.dump(class_labels, f, indent=2)
    
    return model


def main():
    print("\n" + "=" * 60)
    print("   CHART PATTERN RECOGNITION — MULTI-HEAD TRAINING")
    print("=" * 60)

    candlestick_dir = os.path.join(DATA_DIR, 'candlestick')
    chart_dir = os.path.join(DATA_DIR, 'chart_patterns')

    if not os.path.exists(candlestick_dir) or not os.path.exists(chart_dir):
        print("\n  ERROR: Data not found. Run 'python data_generator/generate_all.py' first.")
        sys.exit(1)

    # Train candlestick model
    print("\n  [1/2] Training Candlestick Detection Model...")
    train_model('candlestick_model', candlestick_dir, MODEL_DIR)

    # Train chart pattern model
    print("\n  [2/2] Training Chart Pattern Detection Model...")
    train_model('chart_pattern_model', chart_dir, MODEL_DIR)

    print("\n" + "=" * 60)
    print("   ALL MODELS TRAINED SUCCESSFULLY!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
