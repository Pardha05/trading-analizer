
import tensorflow as tf
import os
import json

MODEL_DIR = r'c:\Users\vardi\Desktop\ABS\saved_models'

def check_model_outputs(filename):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        print(f"\n[SKIP] {filename} not found.")
        return
        
    print(f"\n[CHECK] Inspecting: {filename}")
    try:
        model = tf.keras.models.load_model(path)
        print(f"  Total outputs: {len(model.outputs)}")
        for i, out in enumerate(model.outputs):
            print(f"    Output {i}: {out.name}, shape={out.shape}")
        
        # Test dummy prediction
        import numpy as np
        dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        preds = model.predict(dummy_input, verbose=0)
        
        if isinstance(preds, list):
            print(f"  Inference return type: list (length {len(preds)})")
            for i, p in enumerate(preds):
                print(f"    - Head {i} shape: {p.shape}")
        else:
            print(f"  Inference return type: {type(preds)} (shape {preds.shape})")
            
    except Exception as e:
        print(f"  ERROR checking {filename}: {e}")

if __name__ == "__main__":
    files = [f for f in os.listdir(MODEL_DIR) if f.endswith(('.h5', '.keras'))]
    print(f"Found {len(files)} model files.")
    for f in files:
        check_model_outputs(f)
