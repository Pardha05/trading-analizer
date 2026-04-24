"""
Chart Pattern Predictor using YOLOv8 (foduucom/stockmarket-pattern-detection-yolov8)
Pre-trained model from HuggingFace — no training required.
"""
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

# ─── PyTorch 2.6+ Compatibility Patch ────────────────────────────────────────
import torch
original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load

# ─── YOLOv8 via ultralytics (standard) and huggingface_hub for downloading ───
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import easyocr
import re
import os

# ─── Pattern Metadata ─────────────────────────────────────────────────────────
# ... (rest of the metadata)
PATTERN_TYPE_MAP = {
    'Head and shoulders bottom': 'Bullish Reversal',
    'Head and shoulders top':    'Bearish Reversal',
    'M_Head':                    'Bearish Reversal',   # Double Top
    'W_Bottom':                  'Bullish Reversal',   # Double Bottom
    'Triangle':                  'Neutral',
    'StockLine':                 'Neutral',
}

PATTERN_DISPLAY_NAMES = {
    'Head and shoulders bottom': 'Inv. Head & Shoulders',
    'Head and shoulders top':    'Head & Shoulders',
    'M_Head':                    'Double Top (M-Head)',
    'W_Bottom':                  'Double Bottom (W)',
    'Triangle':                  'Triangle',
    'StockLine':                 'Stock Line',
}

PATTERN_COLORS = {
    'Bullish Reversal':    '#00c853',   # Green
    'Bearish Reversal':    '#ff1744',   # Red
    'Neutral':             '#00e5ff',   # Cyan
}


class PatternPredictor:
    """
    Loads foduucom/stockmarket-pattern-detection-yolov8 from HuggingFace
    and predicts chart patterns from uploaded images.
    Automates price extraction via EasyOCR.
    """

    def __init__(self):
        print("Initializing AI Pattern Predictor (YOLOv8 + EasyOCR)...")
        
        # 1. Download model file directly from HuggingFace Hub
        model_path = hf_hub_download(
            repo_id="foduucom/stockmarket-pattern-detection-yolov8", 
            filename="model.pt"
        )
        
        # 2. Load the model using standard Ultralytics YOLO
        self.model = YOLO(model_path)
        self.model.overrides['conf'] = 0.25
        self.model.overrides['iou']  = 0.45
        self.model.overrides['max_det'] = 15
        
        print("  Initializing EasyOCR Reader...")
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        
        print("  Models loaded successfully!")

        self.candlestick_model = True
        self.chart_model = True

    def predict(self, image_path=None, image_bytes=None, top_k=5, threshold=0.25, recent_focus=True):
        if image_path:
            img_pil = Image.open(image_path).convert('RGB')
        elif image_bytes:
            img_pil = Image.open(BytesIO(image_bytes)).convert('RGB')
        else:
            return {'error': 'No image provided'}

        results = {
            'candlestick_patterns': [],
            'chart_patterns':       [],
            'recommendation':       'HOLD / WAIT',
            'sentiment':            'Neutral',
            'annotated_image':      None,
            'trigger_price':        0,
            'stop_loss':           0,
            'target_price':        0,
            'current_price':       0
        }

        # 1. Detect Patterns
        detections = self._run_yolo(img_pil, threshold)
        if recent_focus:
            w, h = img_pil.size
            recent_crop = img_pil.crop((w // 2, 0, w, h))
            recent_detections = self._run_yolo(recent_crop, threshold, offset_x=0.5)
            detections = self._merge_detections(detections, recent_detections)

        detections.sort(key=lambda x: x['confidence'], reverse=True)
        detections = detections[:top_k]
        results['chart_patterns'] = detections

        # 2. Extract Prices via OCR
        price_map = self._extract_price_map(img_pil)
        
        # 3. Finalize and Calculate Trade Setup
        self._finalize_results(results, img_pil, price_map)

        return results

    def _extract_price_map(self, img_pil):
        """Extract prices from the right axis and return a Y -> Price mapping function."""
        w, h = img_pil.size
        # Crop right 15% where price axis usually resides
        axis_crop = img_pil.crop((int(w * 0.85), 0, w, h))
        img_np = np.array(axis_crop)
        
        ocr_results = self.reader.readtext(img_np)
        
        price_points = []
        for (bbox, text, prob) in ocr_results:
            # Clean text to find numbers (e.g. "1,234.50" -> 1234.5)
            clean_text = re.sub(r'[^\d.]', '', text)
            if not clean_text: continue
            try:
                price = float(clean_text)
                # Y coordinate is center of OCR box
                y_center = (bbox[0][1] + bbox[2][1]) / 2
                price_points.append({'y': y_center / h, 'price': price})
            except ValueError:
                continue

        if len(price_points) < 2:
            return None # Not enough points to interpolate
            
        # Sort by Y
        price_points.sort(key=lambda x: x['y'])
        return price_points

    def _get_price_at_y(self, y, price_map):
        """Interpolate price at a given Y coordinate (0-1)."""
        if not price_map: return 0
        
        # If outside range, use nearest
        if y <= price_map[0]['y']: return price_map[0]['price']
        if y >= price_map[-1]['y']: return price_map[-1]['price']
        
        # Find segment
        for i in range(len(price_map) - 1):
            p1 = price_map[i]
            p2 = price_map[i+1]
            if p1['y'] <= y <= p2['y']:
                # Linear interpolation: P = P1 + (P2 - P1) * (Y - Y1) / (Y2 - Y1)
                ratio = (y - p1['y']) / (p2['y'] - p1['y'])
                return p1['price'] + (p2['price'] - p1['price']) * ratio
        return 0

    def _run_yolo(self, img_pil, threshold, offset_x=0.0):
        img_np = np.array(img_pil)
        yolo_results = self.model.predict(source=img_np, verbose=False)
        detections = []
        for result in yolo_results:
            if result.boxes is None: continue
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence < threshold: continue
                class_id   = int(box.cls[0])
                class_name = self.model.names[class_id]
                x1, y1, x2, y2 = box.xyxyn[0].tolist()
                if offset_x > 0:
                    x1 = offset_x + (x1 * (1.0 - offset_x))
                    x2 = offset_x + (x2 * (1.0 - offset_x))
                detections.append({
                    'pattern':      class_name,
                    'display_name': PATTERN_DISPLAY_NAMES.get(class_name, class_name),
                    'type':         PATTERN_TYPE_MAP.get(class_name, 'Neutral'),
                    'confidence':   round(confidence * 100, 2),
                    'bbox':         [x1, y1, x2, y2],
                })
        return detections

    def _merge_detections(self, base, extra):
        merged = list(base)
        for det in extra:
            is_dup = any(d['pattern'] == det['pattern'] and abs(d['confidence'] - det['confidence']) < 5 for d in merged)
            if not is_dup: merged.append(det)
        return merged

    def _finalize_results(self, results, img_pil, price_map):
        annotated_img  = img_pil.copy()
        bullish_score  = 0.0
        bearish_score  = 0.0

        best_pattern = None
        if results['chart_patterns']:
            best_pattern = results['chart_patterns'][0]

        for p in results['chart_patterns']:
            conf_weight = p['confidence'] / 100.0
            if 'Bullish' in p['type']: bullish_score += conf_weight
            if 'Bearish' in p['type']: bearish_score += conf_weight
            color = PATTERN_COLORS.get(p['type'], '#00e5ff')
            label = f"{p['display_name']} ({p['confidence']}%)"
            annotated_img = self._draw_bbox(annotated_img, p['bbox'], label=label, color=color)

        # Recommendation
        if bullish_score > bearish_score + 0.15:
            results['sentiment'] = 'Bullish'
            results['recommendation'] = 'Strong BUY' if bullish_score > 1.2 else 'BUY'
        elif bearish_score > bullish_score + 0.15:
            results['sentiment'] = 'Bearish'
            results['recommendation'] = 'Strong SELL' if bearish_score > 1.2 else 'SELL'
        else:
            results['sentiment'] = 'Neutral'
            results['recommendation'] = 'HOLD / WAIT'

        # Calculate Trading Prices
        if price_map and best_pattern:
            x1, y1, x2, y2 = best_pattern['bbox']
            price_top = self._get_price_at_y(y1, price_map)
            price_bot = self._get_price_at_y(y2, price_map)
            
            if results['sentiment'] == 'Bullish':
                results['trigger_price'] = price_top # Break above top
                results['stop_loss']     = price_bot # Below bottom
            elif results['sentiment'] == 'Bearish':
                results['trigger_price'] = price_bot # Break below bottom
                results['stop_loss']     = price_top # Above top
            
            # Target 1:2
            risk = abs(results['trigger_price'] - results['stop_loss'])
            if results['sentiment'] == 'Bullish':
                results['target_price'] = results['trigger_price'] + (risk * 2)
            else:
                results['target_price'] = results['trigger_price'] - (risk * 2)
                
            # Generate Advanced Scenarios
            t = results['trigger_price']
            sl = results['stop_loss']
            tg = results['target_price']
            
            if results['sentiment'] == 'Bullish':
                results['pre_trade'] = [
                    {
                        "header": "🔥 SCENARIO 1 — PULLBACK BUY (BEST SETUP)",
                        "condition": f"Price pulls back to {sl + (risk*0.3):.1f}–{sl + (risk*0.5):.1f} zone",
                        "entry": sl + (risk*0.4),
                        "sl": sl,
                        "target": tg,
                        "note": "This is your A+ setup"
                    },
                    {
                        "header": "🔥 SCENARIO 2 — BREAKOUT BUY (CONTINUATION)",
                        "condition": f"Price breaks {t:.1f} clean with momentum",
                        "entry": t + (risk*0.05),
                        "sl": t - (risk*0.2),
                        "target": tg + (risk*0.5),
                        "note": "Only take if strong candle + volume"
                    },
                    {
                        "header": "⚠️ SCENARIO 3 — REVERSAL SELL (RARE)",
                        "condition": f"Price breaks below {sl:.1f} and sustains",
                        "entry": sl - (risk*0.05),
                        "sl": sl + (risk*0.2),
                        "target": sl - (risk*2),
                        "note": "Trend failure reversal"
                    }
                ]
                results['post_trade'] = [
                    {"title": f"Scenario 1: Breaks {t:.1f} zone", "points": ["Strong bullish confirmation", "Hold position", f"Trail SL to {(t+sl)/2:.1f}", f"New target: {tg:.1f} \u2192 {tg+risk:.1f}"]},
                    {"title": f"Scenario 2: Struggles near {t:.1f}", "points": ["Weak breakout / rejection", f"If it fails \u2192 Exit around {t:.1f}", "Don't wait for SL"]},
                    {"title": f"Scenario 3: Falls below {t:.1f}", "points": ["Momentum gone", f"Exit immediately (don't wait for {sl:.1f} SL)"]}
                ]
            else:
                results['pre_trade'] = [
                    {
                        "header": "🔥 SCENARIO 1 — PULLBACK SELL (BEST SETUP)",
                        "condition": f"Price pulls back to {sl - (risk*0.3):.1f}–{sl - (risk*0.5):.1f} zone",
                        "entry": sl - (risk*0.4),
                        "sl": sl,
                        "target": tg,
                        "note": "This is your A+ setup"
                    },
                    {
                        "header": "🔥 SCENARIO 2 — BREAKDOWN SELL (CONTINUATION)",
                        "condition": f"Price breaks {t:.1f} clean with momentum",
                        "entry": t - (risk*0.05),
                        "sl": t + (risk*0.2),
                        "target": tg - (risk*0.5),
                        "note": "Only take if strong candle + volume"
                    },
                    {
                        "header": "⚠️ SCENARIO 3 — REVERSAL BUY (RARE)",
                        "condition": f"Price reclaims {sl:.1f} and sustains above",
                        "entry": sl + (risk*0.05),
                        "sl": sl - (risk*0.2),
                        "target": sl + (risk*2),
                        "note": "Reversal if breakdown fails"
                    }
                ]
                results['post_trade'] = [
                    {"title": f"Scenario 1: Breaks below {t:.1f} zone", "points": ["Strong bearish confirmation", "Hold short position", f"Trail SL to {(t+sl)/2:.1f}", f"New target: {tg:.1f} \u2192 {tg-risk:.1f}"]},
                    {"title": f"Scenario 2: Struggles near {t:.1f}", "points": ["Weak breakdown / support", f"If it fails \u2192 Exit around {t:.1f}", "Don't wait for SL"]},
                    {"title": f"Scenario 3: Rises above {t:.1f}", "points": ["Momentum gone", f"Exit immediately (don't wait for {sl:.1f} SL)"]}
                ]
        
        # Base64 encode
        buffered = BytesIO()
        annotated_img.save(buffered, format='PNG')
        results['annotated_image'] = base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _draw_bbox(self, image, bbox, label=None, color='#00e5ff'):
        draw = ImageDraw.Draw(image, 'RGBA')
        w, h = image.size
        left, top, right, bottom = bbox[0]*w, bbox[1]*h, bbox[2]*w, bbox[3]*h
        draw.rectangle([left, top, right, bottom], outline=color, width=3)
        if label:
            draw.text((left, top - 20), label, fill=color)
        return image
