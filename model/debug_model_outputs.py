
import tensorflow as tf
import os

project_root = r'c:\Users\vardi\Desktop\ABS'
model_dir = os.path.join(project_root, 'saved_models')

models = [
    'candlestick_model.keras',
    'chart_pattern_model.keras'
]

for model_name in models:
    model_path = os.path.join(model_dir, model_name)
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"Model: {model_name}")
            print(f"  Number of outputs: {len(model.outputs)}")
            for i, out in enumerate(model.outputs):
                print(f"    Output {i}: {out.name}, shape={out.shape}")
        except Exception as e:
            print(f"  Error loading {model_name}: {e}")
    else:
        # Try .h5
        model_path = os.path.join(model_dir, model_name.replace('.keras', '.h5'))
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path)
                print(f"Model: {model_name.replace('.keras', '.h5')}")
                print(f"  Number of outputs: {len(model.outputs)}")
            except Exception as e:
                print(f"  Error loading {model_path}: {e}")
        else:
            print(f"Model {model_name} not found.")
