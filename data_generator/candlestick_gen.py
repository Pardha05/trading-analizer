"""
Candlestick Pattern Synthetic Data Generator
Generates labeled candlestick chart images for 12 pattern classes.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import random
import json

# ─── Configuration ───────────────────────────────────────────────────────────
IMG_SIZE = 224
DPI = 72
NUM_IMAGES_PER_CLASS = 300

CANDLESTICK_PATTERNS = {
    'bullish': [
        'hammer', 'inverted_hammer', 'bullish_engulfing',
        'piercing_line', 'morning_star', 'three_white_soldiers'
    ],
    'bearish': [
        'hanging_man', 'shooting_star', 'bearish_engulfing',
        'dark_cloud_cover', 'three_black_crows'
    ],
    'neutral': ['no_pattern']
}

ALL_PATTERNS = []
for pats in CANDLESTICK_PATTERNS.values():
    ALL_PATTERNS.extend(pats)


# ─── OHLC helpers ────────────────────────────────────────────────────────────
def _rand(lo, hi):
    return random.uniform(lo, hi)


def _make_candle(open_p, close_p, high_ext=None, low_ext=None):
    """Create a single candle dict with OHLC."""
    high = max(open_p, close_p) + (high_ext if high_ext else _rand(0.1, 1.0))
    low = min(open_p, close_p) - (low_ext if low_ext else _rand(0.1, 1.0))
    return {'open': open_p, 'high': high, 'low': low, 'close': close_p}


def _context_candles(n, start_price, trend='down', volatility=1.0):
    """Generate n context candles with a trend."""
    candles = []
    price = start_price
    for _ in range(n):
        if trend == 'down':
            change = -_rand(0.5, 2.0) * volatility
        elif trend == 'up':
            change = _rand(0.5, 2.0) * volatility
        else:
            change = _rand(-1.0, 1.0) * volatility
        open_p = price
        close_p = price + change
        candles.append(_make_candle(open_p, close_p))
        price = close_p
    return candles, price


# ─── Pattern generators ─────────────────────────────────────────────────────
def gen_hammer():
    """Hammer: small body at top, long lower shadow, little/no upper shadow."""
    base = _rand(20, 80)
    candles, price = _context_candles(random.randint(3, 6), base, 'down')
    body = _rand(0.3, 1.0)
    lower_shadow = body * _rand(2.0, 4.0)
    upper_shadow = body * _rand(0, 0.3)
    open_p = price
    close_p = price + body
    candles.append({
        'open': open_p, 'close': close_p,
        'high': close_p + upper_shadow, 'low': open_p - lower_shadow
    })
    return candles, [len(candles) - 1]


def gen_inverted_hammer():
    """Inverted Hammer: small body at bottom, long upper shadow."""
    base = _rand(20, 80)
    candles, price = _context_candles(random.randint(3, 6), base, 'down')
    body = _rand(0.3, 1.0)
    upper_shadow = body * _rand(2.0, 4.0)
    lower_shadow = body * _rand(0, 0.3)
    open_p = price
    close_p = price + body
    candles.append({
        'open': open_p, 'close': close_p,
        'high': close_p + upper_shadow, 'low': open_p - lower_shadow
    })
    return candles, [len(candles) - 1]


def gen_bullish_engulfing():
    """Bullish Engulfing: small bearish candle followed by larger bullish candle."""
    base = _rand(20, 80)
    candles, price = _context_candles(random.randint(3, 5), base, 'down')
    # Small bearish candle
    small_body = _rand(0.5, 1.5)
    c1_open = price
    c1_close = price - small_body
    candles.append(_make_candle(c1_open, c1_close))
    # Large bullish candle engulfing the first
    eng_extra = _rand(0.5, 1.5)
    c2_open = c1_close - eng_extra
    c2_close = c1_open + eng_extra
    candles.append(_make_candle(c2_open, c2_close))
    return candles, [len(candles) - 2, len(candles) - 1]


def gen_piercing_line():
    """Piercing Line: bearish candle, then bullish candle closing above midpoint."""
    base = _rand(20, 80)
    candles, price = _context_candles(random.randint(3, 5), base, 'down')
    body1 = _rand(2.0, 4.0)
    c1_open = price
    c1_close = price - body1
    candles.append(_make_candle(c1_open, c1_close))
    midpoint = (c1_open + c1_close) / 2
    c2_open = c1_close - _rand(0.2, 1.0)
    c2_close = midpoint + _rand(0.3, body1 * 0.4)
    candles.append(_make_candle(c2_open, c2_close))
    return candles, [len(candles) - 2, len(candles) - 1]


def gen_morning_star():
    """Morning Star: big bearish, small body (star), big bullish."""
    base = _rand(20, 80)
    candles, price = _context_candles(random.randint(3, 5), base, 'down')
    # First: large bearish
    body1 = _rand(2.0, 4.0)
    c1_open = price
    c1_close = price - body1
    candles.append(_make_candle(c1_open, c1_close))
    # Second: small body (star)
    star_body = _rand(0.1, 0.5)
    star_open = c1_close - _rand(0.2, 0.8)
    star_close = star_open + star_body * random.choice([-1, 1])
    candles.append(_make_candle(star_open, star_close))
    # Third: large bullish
    c3_open = max(star_open, star_close) + _rand(0.1, 0.5)
    c3_close = c1_open - _rand(0, body1 * 0.3)
    candles.append(_make_candle(c3_open, c3_close))
    return candles, [len(candles) - 3, len(candles) - 2, len(candles) - 1]


def gen_three_white_soldiers():
    """Three White Soldiers: three consecutive bullish candles with higher closes."""
    base = _rand(20, 80)
    candles, price = _context_candles(random.randint(3, 5), base, 'down')
    for _ in range(3):
        body = _rand(1.5, 3.0)
        open_p = price + _rand(-0.3, 0.3)
        close_p = open_p + body
        candles.append(_make_candle(open_p, close_p,
                                     high_ext=_rand(0.05, 0.3),
                                     low_ext=_rand(0.05, 0.3)))
        price = close_p
    return candles, [len(candles) - 3, len(candles) - 2, len(candles) - 1]


def gen_hanging_man():
    """Hanging Man: same shape as hammer but after uptrend."""
    base = _rand(20, 60)
    candles, price = _context_candles(random.randint(3, 6), base, 'up')
    body = _rand(0.3, 1.0)
    lower_shadow = body * _rand(2.0, 4.0)
    upper_shadow = body * _rand(0, 0.3)
    open_p = price
    close_p = price - body
    candles.append({
        'open': open_p, 'close': close_p,
        'high': open_p + upper_shadow, 'low': close_p - lower_shadow
    })
    return candles, [len(candles) - 1]


def gen_shooting_star():
    """Shooting Star: same shape as inverted hammer but after uptrend."""
    base = _rand(20, 60)
    candles, price = _context_candles(random.randint(3, 6), base, 'up')
    body = _rand(0.3, 1.0)
    upper_shadow = body * _rand(2.0, 4.0)
    lower_shadow = body * _rand(0, 0.3)
    open_p = price
    close_p = price - body
    candles.append({
        'open': open_p, 'close': close_p,
        'high': open_p + upper_shadow, 'low': close_p - lower_shadow
    })
    return candles, [len(candles) - 1]


def gen_bearish_engulfing():
    """Bearish Engulfing: small bullish candle followed by larger bearish candle."""
    base = _rand(20, 60)
    candles, price = _context_candles(random.randint(3, 5), base, 'up')
    small_body = _rand(0.5, 1.5)
    c1_open = price
    c1_close = price + small_body
    candles.append(_make_candle(c1_open, c1_close))
    eng_extra = _rand(0.5, 1.5)
    c2_open = c1_close + eng_extra
    c2_close = c1_open - eng_extra
    candles.append(_make_candle(c2_open, c2_close))
    return candles, [len(candles) - 2, len(candles) - 1]


def gen_dark_cloud_cover():
    """Dark Cloud Cover: bullish candle, then bearish candle closing below midpoint."""
    base = _rand(20, 60)
    candles, price = _context_candles(random.randint(3, 5), base, 'up')
    body1 = _rand(2.0, 4.0)
    c1_open = price
    c1_close = price + body1
    candles.append(_make_candle(c1_open, c1_close))
    midpoint = (c1_open + c1_close) / 2
    c2_open = c1_close + _rand(0.2, 1.0)
    c2_close = midpoint - _rand(0.3, body1 * 0.4)
    candles.append(_make_candle(c2_open, c2_close))
    return candles, [len(candles) - 2, len(candles) - 1]


def gen_three_black_crows():
    """Three Black Crows: three consecutive bearish candles with lower closes."""
    base = _rand(40, 80)
    candles, price = _context_candles(random.randint(3, 5), base, 'up')
    for _ in range(3):
        body = _rand(1.5, 3.0)
        open_p = price - _rand(-0.3, 0.3)
        close_p = open_p - body
        candles.append(_make_candle(open_p, close_p,
                                     high_ext=_rand(0.05, 0.3),
                                     low_ext=_rand(0.05, 0.3)))
        price = close_p
    return candles, [len(candles) - 3, len(candles) - 2, len(candles) - 1]


def gen_no_pattern():
    """Random candles with no specific pattern."""
    base = _rand(20, 80)
    n = random.randint(5, 10)
    candles, _ = _context_candles(n, base, random.choice(['up', 'down', 'flat']),
                                  volatility=_rand(0.5, 1.5))
    return candles, []


# ─── Map pattern names to generators ────────────────────────────────────────
GENERATORS = {
    'hammer': gen_hammer,
    'inverted_hammer': gen_inverted_hammer,
    'bullish_engulfing': gen_bullish_engulfing,
    'piercing_line': gen_piercing_line,
    'morning_star': gen_morning_star,
    'three_white_soldiers': gen_three_white_soldiers,
    'hanging_man': gen_hanging_man,
    'shooting_star': gen_shooting_star,
    'bearish_engulfing': gen_bearish_engulfing,
    'dark_cloud_cover': gen_dark_cloud_cover,
    'three_black_crows': gen_three_black_crows,
    'no_pattern': gen_no_pattern,
}


# ─── Rendering ───────────────────────────────────────────────────────────────
def render_candlestick_chart(candles, pattern_indices, filepath):
    """Render a list of OHLC candle dicts to a PNG image and return normalized bbox."""
    fig_size = IMG_SIZE / DPI
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=DPI)

    bg_color = random.choice(['#1a1a2e', '#0f0f1a', '#1e1e2f', '#121220', '#0d1117'])
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    for i, c in enumerate(candles):
        is_bullish = c['close'] >= c['open']
        color = '#00c853' if is_bullish else '#ff1744'
        body_bottom = min(c['open'], c['close'])
        body_height = abs(c['close'] - c['open'])

        # Wick
        ax.plot([i, i], [c['low'], c['high']], color=color, linewidth=0.8)
        # Body
        rect = mpatches.FancyBboxPatch(
            (i - 0.3, body_bottom), 0.6, max(body_height, 0.05),
            boxstyle="round,pad=0.02", facecolor=color, edgecolor=color
        )
        ax.add_patch(rect)

    ax.set_xlim(-0.7, len(candles) - 0.3)
    all_lows = [c['low'] for c in candles]
    all_highs = [c['high'] for c in candles]
    min_low = min(all_lows)
    max_high = max(all_highs)
    margin = (max_high - min_low) * 0.1
    ax.set_ylim(min_low - margin, max_high + margin)
    ax.axis('off')

    plt.tight_layout(pad=0.1)
    fig.savefig(filepath, dpi=DPI, bbox_inches='tight',
                facecolor=fig.get_facecolor(), pad_inches=0.02)
    plt.close(fig)

    # Resize to exact 224x224
    img = Image.open(filepath)
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img.save(filepath)

    # Calculate normalized bounding box
    if not pattern_indices:
        return [0, 0, 0, 0]

    # Pattern x boundaries
    x_min_idx = min(pattern_indices) - 0.5
    x_max_idx = max(pattern_indices) + 0.5
    
    # Pattern y boundaries
    p_low = min(candles[i]['low'] for i in pattern_indices)
    p_high = max(candles[i]['high'] for i in pattern_indices)
    
    # Normalize
    xlim_min, xlim_max = -0.7, len(candles) - 0.3
    ylim_min, ylim_max = min_low - margin, max_high + margin
    
    norm_xmin = (x_min_idx - xlim_min) / (xlim_max - xlim_min)
    norm_xmax = (x_max_idx - xlim_min) / (xlim_max - xlim_min)
    norm_ymin = (p_low - ylim_min) / (ylim_max - ylim_min)
    norm_ymax = (p_high - ylim_min) / (ylim_max - ylim_min)
    
    # Flip Y because image coordinates start from top
    return [
        max(0, float(norm_xmin)),
        max(0, float(1.0 - norm_ymax)),
        min(1.0, float(norm_xmax)),
        min(1.0, float(1.0 - norm_ymin))
    ]


# ─── Main generation function ────────────────────────────────────────────────
def generate_candlestick_data(output_dir):
    """Generate all candlestick pattern images."""
    print("=" * 60)
    print("  Generating Candlestick Pattern Images")
    print("=" * 60)

    for pattern_name in ALL_PATTERNS:
        pattern_dir = os.path.join(output_dir, pattern_name)
        os.makedirs(pattern_dir, exist_ok=True)

        gen_func = GENERATORS[pattern_name]
        print(f"\n  [{pattern_name}] Generating {NUM_IMAGES_PER_CLASS} images...")

        for i in range(NUM_IMAGES_PER_CLASS):
            candles, pattern_indices = gen_func()
            base_name = f"{pattern_name}_{i:04d}"
            filepath = os.path.join(pattern_dir, f"{base_name}.png")
            bbox = render_candlestick_chart(candles, pattern_indices, filepath)
            
            # Save bbox to json
            with open(os.path.join(pattern_dir, f"{base_name}.json"), 'w') as f:
                json.dump({"class": pattern_name, "bbox": bbox}, f)

            if (i + 1) % 50 == 0:
                print(f"    ... {i + 1}/{NUM_IMAGES_PER_CLASS}")

        print(f"  [{pattern_name}] Done — {NUM_IMAGES_PER_CLASS} images saved.")

    print(f"\n{'=' * 60}")
    print(f"  Candlestick data generation complete!")
    print(f"  Total: {len(ALL_PATTERNS) * NUM_IMAGES_PER_CLASS} images")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    output = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'candlestick')
    generate_candlestick_data(output)
