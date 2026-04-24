
import tensorflow as tf
import os

model_path = r'c:\Users\vardi\Desktop\ABS\saved_models\candlestick_model.keras'
if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model: {model_path}")
        print(f"Number of outputs: {len(model.outputs)}")
        for i, out in enumerate(model.outputs):
            print(f"  Output {i}: {out.name}, shape={out.shape}")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print("Model file not found.")
