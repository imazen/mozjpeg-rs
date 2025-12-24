#!/usr/bin/env python3
"""
Generate Pareto front plots from benchmark results.

Usage:
    python plot_pareto.py benchmark_results.csv

Outputs:
    - pareto_ssimulacra2.svg: SSIMULACRA2 vs BPP
    - pareto_dssim.svg: DSSIM vs BPP
    - pareto_combined.svg: Both metrics side by side
"""

import sys
import csv
from collections import defaultdict

def load_csv(path):
    """Load benchmark results CSV."""
    results = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'corpus': row['corpus'],
                'image': row['image'],
                'encoder': row['encoder'],
                'quality': int(row['quality']),
                'file_size': int(row['file_size']),
                'bpp': float(row['bpp']),
                'ssimulacra2': float(row['ssimulacra2']),
                'dssim': float(row['dssim']),
            })
    return results

def aggregate_by_quality(results):
    """Aggregate results by encoder and quality level."""
    agg = defaultdict(lambda: defaultdict(list))

    for r in results:
        key = (r['encoder'], r['quality'])
        agg[key]['bpp'].append(r['bpp'])
        agg[key]['ssimulacra2'].append(r['ssimulacra2'])
        agg[key]['dssim'].append(r['dssim'])

    aggregated = []
    for (encoder, quality), values in agg.items():
        aggregated.append({
            'encoder': encoder,
            'quality': quality,
            'bpp': sum(values['bpp']) / len(values['bpp']),
            'ssimulacra2': sum(values['ssimulacra2']) / len(values['ssimulacra2']),
            'dssim': sum(values['dssim']) / len(values['dssim']),
        })

    return aggregated

def generate_svg(data, metric, title, ylabel, lower_is_better=False):
    """Generate SVG Pareto front plot."""
    rust_data = sorted([d for d in data if d['encoder'] == 'rust'], key=lambda x: x['bpp'])
    c_data = sorted([d for d in data if d['encoder'] == 'c'], key=lambda x: x['bpp'])

    # Determine plot bounds
    all_bpp = [d['bpp'] for d in data]
    all_metric = [d[metric] for d in data]

    min_bpp, max_bpp = min(all_bpp), max(all_bpp)
    min_metric, max_metric = min(all_metric), max(all_metric)

    # Add padding
    bpp_range = max_bpp - min_bpp
    metric_range = max_metric - min_metric
    min_bpp -= bpp_range * 0.1
    max_bpp += bpp_range * 0.1
    min_metric -= metric_range * 0.1
    max_metric += metric_range * 0.1

    # SVG dimensions
    width, height = 600, 400
    margin = {'top': 40, 'right': 120, 'bottom': 60, 'left': 80}
    plot_width = width - margin['left'] - margin['right']
    plot_height = height - margin['top'] - margin['bottom']

    def scale_x(v):
        return margin['left'] + (v - min_bpp) / (max_bpp - min_bpp) * plot_width

    def scale_y(v):
        if lower_is_better:
            return margin['top'] + (v - min_metric) / (max_metric - min_metric) * plot_height
        else:
            return margin['top'] + (1 - (v - min_metric) / (max_metric - min_metric)) * plot_height

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">')
    svg.append('<style>')
    svg.append('  .title { font: bold 16px sans-serif; }')
    svg.append('  .axis-label { font: 12px sans-serif; }')
    svg.append('  .legend { font: 12px sans-serif; }')
    svg.append('  .grid { stroke: #e0e0e0; stroke-width: 1; }')
    svg.append('  .rust-line { stroke: #e74c3c; stroke-width: 2; fill: none; }')
    svg.append('  .c-line { stroke: #3498db; stroke-width: 2; fill: none; }')
    svg.append('  .rust-point { fill: #e74c3c; }')
    svg.append('  .c-point { fill: #3498db; }')
    svg.append('</style>')

    # Title
    svg.append(f'<text x="{width/2}" y="25" text-anchor="middle" class="title">{title}</text>')

    # Axes
    svg.append(f'<line x1="{margin["left"]}" y1="{height - margin["bottom"]}" '
               f'x2="{width - margin["right"]}" y2="{height - margin["bottom"]}" stroke="black"/>')
    svg.append(f'<line x1="{margin["left"]}" y1="{margin["top"]}" '
               f'x2="{margin["left"]}" y2="{height - margin["bottom"]}" stroke="black"/>')

    # X axis label
    svg.append(f'<text x="{width/2}" y="{height - 15}" text-anchor="middle" class="axis-label">Bits per Pixel (BPP)</text>')

    # Y axis label
    svg.append(f'<text x="20" y="{height/2}" text-anchor="middle" class="axis-label" '
               f'transform="rotate(-90 20 {height/2})">{ylabel}</text>')

    # Grid lines and ticks
    for i in range(5):
        bpp = min_bpp + i * (max_bpp - min_bpp) / 4
        x = scale_x(bpp)
        svg.append(f'<line x1="{x}" y1="{margin["top"]}" x2="{x}" y2="{height - margin["bottom"]}" class="grid"/>')
        svg.append(f'<text x="{x}" y="{height - margin["bottom"] + 15}" text-anchor="middle" class="axis-label">{bpp:.2f}</text>')

        val = min_metric + i * (max_metric - min_metric) / 4
        y = scale_y(val)
        svg.append(f'<line x1="{margin["left"]}" y1="{y}" x2="{width - margin["right"]}" y2="{y}" class="grid"/>')
        svg.append(f'<text x="{margin["left"] - 10}" y="{y + 4}" text-anchor="end" class="axis-label">{val:.4f}</text>')

    # Plot lines
    if rust_data:
        path = 'M ' + ' L '.join([f'{scale_x(d["bpp"])},{scale_y(d[metric])}' for d in rust_data])
        svg.append(f'<path d="{path}" class="rust-line"/>')
        for d in rust_data:
            svg.append(f'<circle cx="{scale_x(d["bpp"])}" cy="{scale_y(d[metric])}" r="4" class="rust-point"/>')

    if c_data:
        path = 'M ' + ' L '.join([f'{scale_x(d["bpp"])},{scale_y(d[metric])}' for d in c_data])
        svg.append(f'<path d="{path}" class="c-line"/>')
        for d in c_data:
            svg.append(f'<circle cx="{scale_x(d["bpp"])}" cy="{scale_y(d[metric])}" r="4" class="c-point"/>')

    # Legend
    legend_x = width - margin['right'] + 10
    legend_y = margin['top'] + 20
    svg.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="100" height="50" fill="white" stroke="#ccc"/>')
    svg.append(f'<circle cx="{legend_x + 15}" cy="{legend_y + 5}" r="4" class="rust-point"/>')
    svg.append(f'<text x="{legend_x + 25}" y="{legend_y + 9}" class="legend">Rust</text>')
    svg.append(f'<circle cx="{legend_x + 15}" cy="{legend_y + 25}" r="4" class="c-point"/>')
    svg.append(f'<text x="{legend_x + 25}" y="{legend_y + 29}" class="legend">C mozjpeg</text>')

    svg.append('</svg>')

    return '\n'.join(svg)

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_pareto.py benchmark_results.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    results = load_csv(csv_path)
    data = aggregate_by_quality(results)

    # Generate SSIMULACRA2 plot (higher is better)
    svg = generate_svg(data, 'ssimulacra2',
                       'mozjpeg-oxide vs C mozjpeg: SSIMULACRA2',
                       'SSIMULACRA2 Score (higher is better)',
                       lower_is_better=False)
    with open('pareto_ssimulacra2.svg', 'w') as f:
        f.write(svg)
    print("Generated: pareto_ssimulacra2.svg")

    # Generate DSSIM plot (lower is better)
    svg = generate_svg(data, 'dssim',
                       'mozjpeg-oxide vs C mozjpeg: DSSIM',
                       'DSSIM (lower is better)',
                       lower_is_better=True)
    with open('pareto_dssim.svg', 'w') as f:
        f.write(svg)
    print("Generated: pareto_dssim.svg")

    # Print summary table
    print("\nSummary by Quality Level:")
    print("-" * 80)
    print(f"{'Q':>5} {'Rust BPP':>10} {'C BPP':>10} {'BPP Δ%':>10} {'Rust SSIM2':>12} {'C SSIM2':>12} {'SSIM2 Δ':>10}")
    print("-" * 80)

    qualities = sorted(set(d['quality'] for d in data))
    for q in qualities:
        rust = next((d for d in data if d['encoder'] == 'rust' and d['quality'] == q), None)
        c = next((d for d in data if d['encoder'] == 'c' and d['quality'] == q), None)
        if rust and c:
            bpp_diff = (rust['bpp'] - c['bpp']) / c['bpp'] * 100
            ssim_diff = rust['ssimulacra2'] - c['ssimulacra2']
            print(f"{q:>5} {rust['bpp']:>10.4f} {c['bpp']:>10.4f} {bpp_diff:>+9.2f}% "
                  f"{rust['ssimulacra2']:>12.4f} {c['ssimulacra2']:>12.4f} {ssim_diff:>+10.4f}")

if __name__ == '__main__':
    main()
