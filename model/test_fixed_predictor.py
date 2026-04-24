
import os
import sys
import numpy as np
from PIL import Image

# Add project root to path
project_root = r'c:\Users\vardi\Desktop\ABS'
sys.path.insert(0, project_root)

from model.predict import PatternPredictor

def test_predictor():
    print("Testing PatternPredictor...")
    predictor = PatternPredictor()
    
    # Create a dummy image
    dummy_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    
    import io
    img_bytes = io.BytesIO()
    dummy_img.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    
    try:
        print("Running prediction...")
        results = predictor.predict(image_bytes=img_bytes)
        print("Success!")
        print("Candlestick patterns:", results['candlestick_patterns'])
        print("Chart patterns:", results['chart_patterns'])
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_predictor()
