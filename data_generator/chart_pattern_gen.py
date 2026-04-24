"""
Chart Pattern Synthetic Data Generator
Generates labeled chart pattern images for 16 pattern classes.
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

CHART_PATTERNS = {
    'bearish': [
        'double_top', 'triple_top', 'head_and_shoulders',
        'descending_triangle', 'bearish_flag', 'bearish_wedge',
        'bearish_pennant', 'inverse_cup_and_handle'
    ],
    'bullish': [
        'double_bottom', 'triple_bottom', 'inverted_head_and_shoulders',
        'ascending_triangle', 'bullish_flag', 'bullish_wedge',
        'cup_and_handle'
    ],
    'neutral': ['no_chart_pattern']
}

ALL_PATTERNS = []
for pats in CHART_PATTERNS.values():
    ALL_PATTERNS.extend(pats)


# ─── Utility functions ──────────────────────────────────────────────────────
def _rand(lo, hi):
    return random.uniform(lo, hi)


def _add_noise(prices, noise_level=0.3):
    """Add random noise to a price series."""
    return [p + random.gauss(0, noise_level) for p in prices]


def _generate_ohlc_from_prices(prices):
    """Convert close prices to OHLC candles."""
    candles = []
    for i in range(len(prices)):
        close = prices[i]
        open_p = prices[i - 1] if i > 0 else close + _rand(-0.5, 0.5)
        high = max(open_p, close) + abs(_rand(0.1, 0.8))
        low = min(open_p, close) - abs(_rand(0.1, 0.8))
        candles.append({'open': open_p, 'high': high, 'low': low, 'close': close})
    return candles


def _trend_series(n, start, end, noise=0.3):
    """Generate a noisy trend from start to end price."""
    base = np.linspace(start, end, n)
    return _add_noise(base.tolist(), noise)


# ─── Chart Pattern Price Series Generators ───────────────────────────────────

def gen_double_top():
    """M shape: rise, peak, dip, peak, drop."""
    base = _rand(30, 50)
    peak = base + _rand(10, 20)
    valley = base + _rand(3, 7)
    n_seg = random.randint(6, 10)
    prices = (
        _trend_series(n_seg, base, peak) +
        _trend_series(n_seg, peak, valley) +
        _trend_series(n_seg, valley, peak + _rand(-1, 1)) +
        _trend_series(n_seg, peak, base - _rand(2, 5))
    )
    return prices, list(range(len(prices)))


def gen_double_bottom():
    """W shape: drop, valley, rise, valley, rise."""
    base = _rand(50, 70)
    valley = base - _rand(10, 20)
    peak = base - _rand(3, 7)
    n_seg = random.randint(6, 10)
    prices = (
        _trend_series(n_seg, base, valley) +
        _trend_series(n_seg, valley, peak) +
        _trend_series(n_seg, peak, valley + _rand(-1, 1)) +
        _trend_series(n_seg, valley, base + _rand(2, 5))
    )
    return prices, list(range(len(prices)))


def gen_triple_top():
    """Three peaks at similar level."""
    base = _rand(30, 50)
    peak = base + _rand(10, 18)
    valley = base + _rand(3, 7)
    n_seg = random.randint(5, 8)
    prices = (
        _trend_series(n_seg, base, peak) +
        _trend_series(n_seg, peak, valley) +
        _trend_series(n_seg, valley, peak + _rand(-1, 1)) +
        _trend_series(n_seg, peak, valley + _rand(-1, 1)) +
        _trend_series(n_seg, valley, peak + _rand(-1, 1)) +
        _trend_series(n_seg, peak, base - _rand(2, 5))
    )
    return prices, list(range(len(prices)))


def gen_triple_bottom():
    """Three valleys at similar level."""
    base = _rand(50, 70)
    valley = base - _rand(10, 18)
    peak = base - _rand(3, 7)
    n_seg = random.randint(5, 8)
    prices = (
        _trend_series(n_seg, base, valley) +
        _trend_series(n_seg, valley, peak) +
        _trend_series(n_seg, peak, valley + _rand(-1, 1)) +
        _trend_series(n_seg, valley, peak + _rand(-1, 1)) +
        _trend_series(n_seg, peak, valley + _rand(-1, 1)) +
        _trend_series(n_seg, valley, base + _rand(2, 5))
    )
    return prices, list(range(len(prices)))


def gen_head_and_shoulders():
    """Left shoulder, head (higher), right shoulder, neckline break."""
    base = _rand(30, 50)
    shoulder = base + _rand(8, 12)
    head = shoulder + _rand(5, 10)
    neck = base + _rand(1, 4)
    n_seg = random.randint(5, 8)
    prices = (
        _trend_series(n_seg, base, shoulder) +
        _trend_series(n_seg, shoulder, neck) +
        _trend_series(n_seg, neck, head) +
        _trend_series(n_seg, head, neck + _rand(-1, 1)) +
        _trend_series(n_seg, neck, shoulder + _rand(-1, 1)) +
        _trend_series(n_seg, shoulder, base - _rand(2, 5))
    )
    return prices, list(range(len(prices)))


def gen_inverted_head_and_shoulders():
    """Inverse of head and shoulders."""
    base = _rand(50, 70)
    shoulder = base - _rand(8, 12)
    head = shoulder - _rand(5, 10)
    neck = base - _rand(1, 4)
    n_seg = random.randint(5, 8)
    prices = (
        _trend_series(n_seg, base, shoulder) +
        _trend_series(n_seg, shoulder, neck) +
        _trend_series(n_seg, neck, head) +
        _trend_series(n_seg, head, neck + _rand(-1, 1)) +
        _trend_series(n_seg, neck, shoulder + _rand(-1, 1)) +
        _trend_series(n_seg, shoulder, base + _rand(2, 5))
    )
    return prices, list(range(len(prices)))


def gen_ascending_triangle():
    """Flat resistance, rising support."""
    base_low = _rand(30, 45)
    resistance = base_low + _rand(15, 25)
    n_waves = random.randint(3, 5)
    n_seg = random.randint(5, 8)
    prices = []
    for i in range(n_waves):
        support = base_low + (resistance - base_low) * (i / (n_waves + 1)) * 0.6
        prices += _trend_series(n_seg, support, resistance + _rand(-1, 0.5))
        if i < n_waves - 1:
            next_support = base_low + (resistance - base_low) * ((i + 1) / (n_waves + 1)) * 0.6
            prices += _trend_series(n_seg, resistance, next_support)
    # Breakout
    prices += _trend_series(n_seg, resistance, resistance + _rand(5, 10))
    return prices, list(range(len(prices)))


def gen_descending_triangle():
    """Flat support, falling resistance."""
    base_high = _rand(55, 75)
    support = base_high - _rand(15, 25)
    n_waves = random.randint(3, 5)
    n_seg = random.randint(5, 8)
    prices = []
    for i in range(n_waves):
        res = base_high - (base_high - support) * (i / (n_waves + 1)) * 0.6
        prices += _trend_series(n_seg, res, support + _rand(-0.5, 1))
        if i < n_waves - 1:
            next_res = base_high - (base_high - support) * ((i + 1) / (n_waves + 1)) * 0.6
            prices += _trend_series(n_seg, support, next_res)
    # Breakdown
    prices += _trend_series(n_seg, support, support - _rand(5, 10))
    return prices, list(range(len(prices)))


def gen_bullish_flag():
    """Sharp rise (pole), then slight downward channel (flag), then continuation."""
    base = _rand(25, 40)
    pole_top = base + _rand(15, 25)
    n_seg = random.randint(5, 8)
    flag_n = random.randint(8, 14)
    # Pole
    prices = _trend_series(n_seg, base, pole_top, noise=0.5)
    # Flag (slight downward consolidation)
    flag_drop = _rand(3, 6)
    prices += _trend_series(flag_n, pole_top, pole_top - flag_drop, noise=0.8)
    # Continuation
    prices += _trend_series(n_seg, pole_top - flag_drop, pole_top + _rand(3, 8), noise=0.5)
    return prices, list(range(len(prices)))


def gen_bearish_flag():
    """Sharp drop (pole), then slight upward channel (flag), then continuation down."""
    base = _rand(60, 80)
    pole_bottom = base - _rand(15, 25)
    n_seg = random.randint(5, 8)
    flag_n = random.randint(8, 14)
    # Pole
    prices = _trend_series(n_seg, base, pole_bottom, noise=0.5)
    # Flag (slight upward consolidation)
    flag_rise = _rand(3, 6)
    prices += _trend_series(flag_n, pole_bottom, pole_bottom + flag_rise, noise=0.8)
    # Continuation
    prices += _trend_series(n_seg, pole_bottom + flag_rise, pole_bottom - _rand(3, 8), noise=0.5)
    return prices, list(range(len(prices)))


def gen_bullish_wedge():
    """Falling wedge -> bullish breakout. Converging downward trendlines."""
    base = _rand(50, 70)
    n_waves = random.randint(4, 6)
    n_seg = random.randint(4, 6)
    prices = []
    high = base
    low = base - _rand(5, 10)
    for i in range(n_waves):
        convergence = 1 - (i / n_waves) * 0.5
        prices += _trend_series(n_seg, high, low)
        high -= _rand(1, 3) * convergence
        low -= _rand(0.5, 1.5) * convergence
        if i < n_waves - 1:
            prices += _trend_series(n_seg, low, high)
    # Bullish breakout
    prices += _trend_series(n_seg * 2, low, base + _rand(5, 12), noise=0.5)
    return prices, list(range(len(prices)))


def gen_bearish_wedge():
    """Rising wedge -> bearish breakdown. Converging upward trendlines."""
    base = _rand(30, 50)
    n_waves = random.randint(4, 6)
    n_seg = random.randint(4, 6)
    prices = []
    low = base
    high = base + _rand(5, 10)
    for i in range(n_waves):
        convergence = 1 - (i / n_waves) * 0.5
        prices += _trend_series(n_seg, low, high)
        low += _rand(0.5, 1.5) * convergence
        high += _rand(1, 3) * convergence
        if i < n_waves - 1:
            prices += _trend_series(n_seg, high, low)
    # Bearish breakdown
    prices += _trend_series(n_seg * 2, high, base - _rand(5, 12), noise=0.5)
    return prices, list(range(len(prices)))


def gen_bearish_pennant():
    """Sharp drop, then symmetrical triangular consolidation, then continuation down."""
    base = _rand(60, 80)
    pole_bottom = base - _rand(15, 25)
    n_seg = random.randint(4, 6)
    prices = _trend_series(n_seg, base, pole_bottom, noise=0.5)
    # Pennant (symmetrical triangle)
    mid = pole_bottom + _rand(3, 6)
    amplitude = _rand(3, 5)
    n_pennant_waves = random.randint(3, 5)
    for i in range(n_pennant_waves):
        decay = 1 - (i / n_pennant_waves) * 0.7
        if i % 2 == 0:
            prices += _trend_series(n_seg, mid - amplitude * decay, mid + amplitude * decay, noise=0.3)
        else:
            prices += _trend_series(n_seg, mid + amplitude * decay, mid - amplitude * decay, noise=0.3)
    # Continuation down
    prices += _trend_series(n_seg, mid, pole_bottom - _rand(5, 10), noise=0.5)
    return prices, list(range(len(prices)))


def gen_cup_and_handle():
    """U-shaped cup followed by small downward handle, then breakout."""
    base = _rand(50, 70)
    cup_depth = _rand(10, 20)
    n_seg = random.randint(6, 10)
    # Left rim
    left_rim = base
    cup_bottom = base - cup_depth
    # Cup (U shape)
    t = np.linspace(0, np.pi, n_seg * 3)
    cup_prices = (left_rim - cup_depth * np.sin(t)).tolist()
    prices = _add_noise(cup_prices, 0.5)
    # Handle (small dip)
    handle_depth = cup_depth * _rand(0.2, 0.4)
    prices += _trend_series(n_seg, base, base - handle_depth, noise=0.3)
    prices += _trend_series(n_seg, base - handle_depth, base + _rand(3, 8), noise=0.4)
    return prices, list(range(len(prices)))


def gen_inverse_cup_and_handle():
    """Inverted U-shape followed by small upward handle, then breakdown."""
    base = _rand(30, 50)
    cup_height = _rand(10, 20)
    n_seg = random.randint(6, 10)
    # Inverted cup (∩ shape)
    t = np.linspace(0, np.pi, n_seg * 3)
    cup_prices = (base + cup_height * np.sin(t)).tolist()
    prices = _add_noise(cup_prices, 0.5)
    # Handle (small rise)
    handle_height = cup_height * _rand(0.2, 0.4)
    prices += _trend_series(n_seg, base, base + handle_height, noise=0.3)
    prices += _trend_series(n_seg, base + handle_height, base - _rand(3, 8), noise=0.4)
    return prices, list(range(len(prices)))


def gen_no_chart_pattern():
    """Random price series with no specific pattern."""
    base = _rand(30, 70)
    n = random.randint(25, 50)
    prices = [base]
    for _ in range(n - 1):
        prices.append(prices[-1] + random.gauss(0, 1))
    return prices, []


# ─── Map pattern names to generators ────────────────────────────────────────
GENERATORS = {
    'double_top': gen_double_top,
    'double_bottom': gen_double_bottom,
    'triple_top': gen_triple_top,
    'triple_bottom': gen_triple_bottom,
    'head_and_shoulders': gen_head_and_shoulders,
    'inverted_head_and_shoulders': gen_inverted_head_and_shoulders,
    'ascending_triangle': gen_ascending_triangle,
    'descending_triangle': gen_descending_triangle,
    'bullish_flag': gen_bullish_flag,
    'bearish_flag': gen_bearish_flag,
    'bullish_wedge': gen_bullish_wedge,
    'bearish_wedge': gen_bearish_wedge,
    'bearish_pennant': gen_bearish_pennant,
    'cup_and_handle': gen_cup_and_handle,
    'inverse_cup_and_handle': gen_inverse_cup_and_handle,
    'no_chart_pattern': gen_no_chart_pattern,
}


# ─── Rendering ───────────────────────────────────────────────────────────────
def render_chart(prices, pattern_indices, filepath, style='candle'):
    """Render a price series as a chart image and return normalized bbox."""
    candles = _generate_ohlc_from_prices(prices)
    fig_size = IMG_SIZE / DPI
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=DPI)

    bg_color = random.choice(['#1a1a2e', '#0f0f1a', '#1e1e2f', '#121220', '#0d1117'])
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    render_type = random.choice(['candle', 'line', 'candle'])  # Bias toward candles

    if render_type == 'candle':
        for i, c in enumerate(candles):
            is_bullish = c['close'] >= c['open']
            color = '#00c853' if is_bullish else '#ff1744'
            body_bottom = min(c['open'], c['close'])
            body_height = abs(c['close'] - c['open'])
            ax.plot([i, i], [c['low'], c['high']], color=color, linewidth=0.6)
            rect = mpatches.FancyBboxPatch(
                (i - 0.3, body_bottom), 0.6, max(body_height, 0.05),
                boxstyle="round,pad=0.02", facecolor=color, edgecolor=color
            )
            ax.add_patch(rect)
        ax.set_xlim(-0.7, len(candles) - 0.3)
    else:
        line_color = random.choice(['#00e5ff', '#76ff03', '#ffd600', '#ff6d00'])
        ax.plot(prices, color=line_color, linewidth=1.2)
        ax.fill_between(range(len(prices)), prices, min(prices) - 2,
                        alpha=0.1, color=line_color)
        ax.set_xlim(0, len(prices) - 1)

    all_vals = [c['low'] for c in candles] + [c['high'] for c in candles]
    min_val = min(all_vals)
    max_val = max(all_vals)
    margin = (max_val - min_val) * 0.1
    ax.set_ylim(min_val - margin, max_val + margin)
    ax.axis('off')

    plt.tight_layout(pad=0.1)
    fig.savefig(filepath, dpi=DPI, bbox_inches='tight',
                facecolor=fig.get_facecolor(), pad_inches=0.02)
    plt.close(fig)

    img = Image.open(filepath)
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img.save(filepath)

    # Calculate normalized bounding box
    if not pattern_indices:
        return [0, 0, 0, 0]

    # Pattern x boundaries
    x_min_idx = min(pattern_indices)
    x_max_idx = max(pattern_indices)
    
    if render_type == 'candle':
        x_min_val = x_min_idx - 0.5
        x_max_val = x_max_idx + 0.5
        xlim_min, xlim_max = -0.7, len(prices) - 0.3
    else:
        x_min_val = x_min_idx
        x_max_val = x_max_idx
        xlim_min, xlim_max = 0, len(prices) - 1
        
    # Pattern y boundaries
    p_prices = [prices[i] for i in pattern_indices]
    p_low = min(p_prices) - (0.5 if render_type == 'candle' else 0)
    p_high = max(p_prices) + (0.5 if render_type == 'candle' else 0)
    
    # Normalize
    ylim_min, ylim_max = min_val - margin, max_val + margin
    
    norm_xmin = (x_min_val - xlim_min) / (xlim_max - xlim_min)
    norm_xmax = (x_max_val - xlim_min) / (xlim_max - xlim_min)
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
def generate_chart_pattern_data(output_dir):
    """Generate all chart pattern images."""
    print("=" * 60)
    print("  Generating Chart Pattern Images")
    print("=" * 60)

    for pattern_name in ALL_PATTERNS:
        pattern_dir = os.path.join(output_dir, pattern_name)
        os.makedirs(pattern_dir, exist_ok=True)

        gen_func = GENERATORS[pattern_name]
        print(f"\n  [{pattern_name}] Generating {NUM_IMAGES_PER_CLASS} images...")

        for i in range(NUM_IMAGES_PER_CLASS):
            prices, pattern_indices = gen_func()
            base_name = f"{pattern_name}_{i:04d}"
            filepath = os.path.join(pattern_dir, f"{base_name}.png")
            bbox = render_chart(prices, pattern_indices, filepath)
            
            # Save bbox to json
            with open(os.path.join(pattern_dir, f"{base_name}.json"), 'w') as f:
                json.dump({"class": pattern_name, "bbox": bbox}, f)

            if (i + 1) % 50 == 0:
                print(f"    ... {i + 1}/{NUM_IMAGES_PER_CLASS}")

        print(f"  [{pattern_name}] Done — {NUM_IMAGES_PER_CLASS} images saved.")

    print(f"\n{'=' * 60}")
    print(f"  Chart pattern data generation complete!")
    print(f"  Total: {len(ALL_PATTERNS) * NUM_IMAGES_PER_CLASS} images")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    output = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'chart_patterns')
    generate_chart_pattern_data(output)
