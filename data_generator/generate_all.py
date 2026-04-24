"""
Master data generation script.
Generates all synthetic training data for both candlestick and chart patterns.
"""
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from data_generator.candlestick_gen import generate_candlestick_data
from data_generator.chart_pattern_gen import generate_chart_pattern_data


def main():
    data_dir = os.path.join(project_root, 'data')

    candlestick_dir = os.path.join(data_dir, 'candlestick')
    chart_dir = os.path.join(data_dir, 'chart_patterns')

    os.makedirs(candlestick_dir, exist_ok=True)
    os.makedirs(chart_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("   CHART PATTERN RECOGNITION — DATA GENERATOR")
    print("=" * 60)

    # Step 1: Candlestick patterns
    generate_candlestick_data(candlestick_dir)

    print("\n")

    # Step 2: Chart patterns
    generate_chart_pattern_data(chart_dir)

    print("\n" + "=" * 60)
    print("   ALL DATA GENERATION COMPLETE!")
    print("=" * 60)
    print(f"   Data saved to: {data_dir}")
    print(f"   Candlestick patterns: {candlestick_dir}")
    print(f"   Chart patterns:       {chart_dir}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
