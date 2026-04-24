
import os
import sys
import json
import base64
from PIL import Image

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from model.predict import PatternPredictor

def verify_localization():
    print("\n" + "="*50)
    print(" PROGRAMMATIC LOCALIZATION VERIFICATION")
    print("="*50)
    
    predictor = PatternPredictor()
    
    # Test cases: (Image Path, Expected Class)
    test_cases = [
        (os.path.join(project_root, 'data', 'candlestick', 'hammer', 'hammer_0000.png'), 'hammer'),
        (os.path.join(project_root, 'data', 'chart_patterns', 'double_top', 'double_top_0000.png'), 'double_top')
    ]
    
    for img_path, expected_class in test_cases:
        print(f"\n[TEST] Processing: {os.path.basename(img_path)}")
        
        if not os.path.exists(img_path):
            print(f"  ERROR: Image not found at {img_path}")
            continue
            
        # Run prediction
        results = predictor.predict(image_path=img_path)
        
        # Check results
        cs_preds = results.get('candlestick_patterns', [])
        cp_preds = results.get('chart_patterns', [])
        annotated_b64 = results.get('annotated_image')
        
        found_expected = False
        all_preds = cs_preds + cp_preds
        
        for p in all_preds:
            pattern = p['pattern']
            conf = p['confidence']
            bbox = p.get('bbox', [0,0,0,0])
            
            print(f"  - Detected: {pattern} ({conf}%)")
            print(f"  - BBox: {bbox}")
            
            if pattern == expected_class:
                found_expected = True
                
        if found_expected:
            print(f"  SUCCESS: Found expected class '{expected_class}'")
        else:
            print(f"  FAILURE: Expected '{expected_class}' not among detections")
            
        if annotated_b64:
            print(f"  SUCCESS: Annotated image generated (size: {len(annotated_b64)} chars)")
            # Save the annotated image to verify manually if needed
            save_path = os.path.join(project_root, f"verify_{expected_class}.png")
            with open(save_path, "wb") as f:
                f.write(base64.b64decode(annotated_b64))
            print(f"  Result saved to: {save_path}")
        else:
            print(f"  FAILURE: No annotated image in output")

if __name__ == "__main__":
    verify_localization()
