#!/usr/bin/env python3
"""
Visualization script for MARL fiber oversubscription microbenchmark with perf counters.
Generates professional plots showing performance and hardware counter impacts.
NORMALIZED by worker count to show true efficiency degradation.
"""

import sys
import os
import subprocess

# Install required packages
def install_packages():
    """Install required packages if not already installed."""
    packages = ['matplotlib', 'pandas', 'numpy']
    
    print("Checking and installing required packages...")
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package} is already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ {package} installed successfully")

# Install packages before importing them
install_packages()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_results(filename='benchmark_results_with_perf.csv'):
    """Load benchmark results from CSV file."""
    if not os.path.exists(filename):
        print(f"Error: {filename} not found!")
        print("Run the microbenchmark first to generate results.")
        sys.exit(1)
    
    df = pd.read_csv(filename)
    
    # Calculate normalized metrics
    df['throughput_per_worker'] = df['throughput'] / df['num_workers']
    df['efficiency_ratio'] = df['throughput_per_worker'] / df['throughput_per_worker'].max()
    df['cycles_per_task'] = df['cpu_cycles'] / (df['num_threads'] * 100)  # 100 iterations per fiber
    df['instructions_per_task'] = df['instructions'] / (df['num_threads'] * 100)
    
    return df

def setup_plot_style():
    """Set up professional plotting style."""
    plt.rcParams.update({
        'font.size': 14, 
        'font.family': 'serif'
    })

def apply_plot_styling(ax, xlabel, ylabel, title, color='#CC0066', add_optimal=True, df=None):
    """Apply consistent styling to all plots."""
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_title(title, fontsize=24, fontweight='bold', pad=15)
    ax.grid(True, axis='y', alpha=0.8, linestyle=':', color='black', linewidth=2.0, zorder=0)
    ax.tick_params(axis='both', labelsize=16)
    
    # Add optimal reference line if requested
    if add_optimal and df is not None:
        optimal_idx = df['oversubscription_ratio'].sub(1.0).abs().idxmin()
        ax.axvline(x=df.loc[optimal_idx, 'oversubscription_ratio'], 
                   color='#060771', linestyle='--', alpha=0.8, 
                   linewidth=2.5, label='Optimal (~1x)', zorder=2)
        legend = ax.legend(frameon=True, fancybox=False, shadow=False,
                          loc='best', fontsize=18, edgecolor='black')
        legend.get_frame().set_linewidth(2.0)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.95)
    
    # Add borders
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(2.5)

def save_figure(filename_base, saved_files):
    """Save figure in multiple formats."""
    for ext in ['png', 'pdf', 'eps']:
        filename = f'{filename_base}.{ext}'
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        saved_files.append(filename)

def create_visualizations(df):
    """Create comprehensive visualization plots."""
    
    setup_plot_style()
    saved_files = []
    
    # Color palette
    colors = {
        'throughput': '#CC0066',        # magenta
        'efficiency': '#8B008B',        # dark magenta
        'switches': '#006666',           # dark teal
        'computation': '#F18F01',        # orange
        'total_time': 'lime',            # lime
        'ipc': '#9B59B6',               # purple
        'cache_miss': '#E74C3C',        # red
        'branch_miss': '#3498DB',       # blue
        'context_switches': '#16A085',  # turquoise
        'cpu_cycles': '#F39C12',        # yellow-orange
        'instructions': '#27AE60',      # green
        'page_faults': '#D35400',       # burnt orange
    }
    
    # ========== Plot 1: Throughput PER WORKER (Normalized - PRIMARY METRIC) ==========
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(df['oversubscription_ratio'], df['throughput_per_worker'], 
            marker='o', linewidth=3, markersize=10, color=colors['throughput'],
            markeredgecolor='black', markeredgewidth=1.5, zorder=3)
    apply_plot_styling(ax, 'Oversubscription Ratio (fibers/workers)', 
                      'Throughput per Worker (tasks/sec/worker)', 
                      'Per-Worker Efficiency vs Oversubscription',
                      colors['throughput'], True, df)
    plt.tight_layout()
    save_figure('01_throughput_per_worker_vs_oversubscription', saved_files)
    plt.close()
    
    # ========== Plot 2: System Efficiency (%) ==========
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(df['oversubscription_ratio'], df['efficiency_ratio'] * 100, 
            marker='o', linewidth=3, markersize=10, color=colors['efficiency'],
            markeredgecolor='black', markeredgewidth=1.5, zorder=3)
    apply_plot_styling(ax, 'Oversubscription Ratio (fibers/workers)', 
                      'System Efficiency (%)', 
                      'System Efficiency Degradation with Oversubscription',
                      colors['efficiency'], True, df)
    ax.set_ylim([0, 105])  # 0-100% range
    ax.axhline(y=100, color='green', linestyle=':', alpha=0.5, linewidth=2, label='100% Efficiency', zorder=1)
    plt.tight_layout()
    save_figure('02_efficiency_vs_oversubscription', saved_files)
    plt.close()
    
    # ========== Plot 3: Absolute Throughput (Reference - shows total work scaling) ==========
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(df['oversubscription_ratio'], df['throughput'], 
            marker='o', linewidth=3, markersize=10, color='gray',
            markeredgecolor='black', markeredgewidth=1.5, zorder=3)
    apply_plot_styling(ax, 'Oversubscription Ratio (fibers/workers)', 
                      'Absolute Throughput (tasks/sec)', 
                      'Absolute Throughput [Reference - Total Work Scales]',
                      'gray', False, df)
    # Add annotation
    ax.text(0.5, 0.95, 'Note: This metric increases because total work scales with fiber count',
            transform=ax.transAxes, fontsize=14, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    plt.tight_layout()
    save_figure('03_absolute_throughput_reference', saved_files)
    plt.close()
    
    # ========== Plot 4: Fiber Switches vs Oversubscription ==========
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(df['oversubscription_ratio'], df['thread_switches'], 
            marker='s', linewidth=3, markersize=10, color=colors['switches'],
            markeredgecolor='black', markeredgewidth=1.5, zorder=3)
    apply_plot_styling(ax, 'Oversubscription Ratio (fibers/workers)', 
                      'Fiber Switches (count)', 
                      'Fiber Switches vs Oversubscription',
                      colors['switches'], False, df)
    plt.tight_layout()
    save_figure('04_fiber_switches_vs_oversubscription', saved_files)
    plt.close()
    
    # ========== Plot 5: Average Computation Time ==========
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(df['oversubscription_ratio'], df['avg_computation_time_ms'], 
            marker='^', linewidth=3, markersize=10, color=colors['computation'],
            markeredgecolor='black', markeredgewidth=1.5, zorder=3)
    apply_plot_styling(ax, 'Oversubscription Ratio (fibers/workers)', 
                      'Avg Computation Time (ms)', 
                      'Computation Time Increase with Oversubscription',
                      colors['computation'], True, df)
    plt.tight_layout()
    save_figure('05_computation_time_vs_oversubscription', saved_files)
    plt.close()
    
    # ========== Plot 6: Total Execution Time ==========
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(df['oversubscription_ratio'], df['total_time_ms'], 
            marker='D', linewidth=3, markersize=10, color=colors['total_time'],
            markeredgecolor='black', markeredgewidth=1.5, zorder=3)
    apply_plot_styling(ax, 'Oversubscription Ratio (fibers/workers)', 
                      'Total Time (ms)', 
                      'Total Execution Time vs Oversubscription',
                      colors['total_time'], False, df)
    plt.tight_layout()
    save_figure('06_total_time_vs_oversubscription', saved_files)
    plt.close()
    
    # ========== Plot 7: IPC (Instructions Per Cycle) ==========
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(df['oversubscription_ratio'], df['ipc'], 
            marker='o', linewidth=3, markersize=10, color=colors['ipc'],
            markeredgecolor='black', markeredgewidth=1.5, zorder=3)
    apply_plot_styling(ax, 'Oversubscription Ratio (fibers/workers)', 
                      'Instructions Per Cycle (IPC)', 
                      'IPC Degradation with Oversubscription',
                      colors['ipc'], True, df)
    plt.tight_layout()
    save_figure('07_ipc_vs_oversubscription', saved_files)
    plt.close()
    
    # ========== Plot 8: Cache Miss Rate ==========
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(df['oversubscription_ratio'], df['cache_miss_rate'], 
            marker='s', linewidth=3, markersize=10, color=colors['cache_miss'],
            markeredgecolor='black', markeredgewidth=1.5, zorder=3)
    apply_plot_styling(ax, 'Oversubscription Ratio (fibers/workers)', 
                      'Cache Miss Rate (%)', 
                      'Cache Pollution from Fiber Oversubscription',
                      colors['cache_miss'], True, df)
    plt.tight_layout()
    save_figure('08_cache_miss_rate_vs_oversubscription', saved_files)
    plt.close()
    
    # ========== Plot 9: Branch Miss Rate ==========
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(df['oversubscription_ratio'], df['branch_miss_rate'], 
            marker='^', linewidth=3, markersize=10, color=colors['branch_miss'],
            markeredgecolor='black', markeredgewidth=1.5, zorder=3)
    apply_plot_styling(ax, 'Oversubscription Ratio (fibers/workers)', 
                      'Branch Miss Rate (%)', 
                      'Branch Predictor Pollution from Oversubscription',
                      colors['branch_miss'], True, df)
    plt.tight_layout()
    save_figure('09_branch_miss_rate_vs_oversubscription', saved_files)
    plt.close()
    
    # ========== Plot 10: Context Switches ==========
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(df['oversubscription_ratio'], df['context_switches'], 
            marker='D', linewidth=3, markersize=10, color=colors['context_switches'],
            markeredgecolor='black', markeredgewidth=1.5, zorder=3)
    apply_plot_styling(ax, 'Oversubscription Ratio (fibers/workers)', 
                      'OS Context Switches (count)', 
                      'Kernel Context Switches vs Oversubscription',
                      colors['context_switches'], False, df)
    plt.tight_layout()
    save_figure('10_context_switches_vs_oversubscription', saved_files)
    plt.close()
    
    # ========== Plot 11: CPU Cycles Per Task (Normalized) ==========
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(df['oversubscription_ratio'], df['cycles_per_task'] / 1e6, 
            marker='o', linewidth=3, markersize=10, color=colors['cpu_cycles'],
            markeredgecolor='black', markeredgewidth=1.5, zorder=3)
    apply_plot_styling(ax, 'Oversubscription Ratio (fibers/workers)', 
                      'CPU Cycles per Task (millions)', 
                      'CPU Cycles per Task vs Oversubscription',
                      colors['cpu_cycles'], True, df)
    plt.tight_layout()
    save_figure('11_cycles_per_task_vs_oversubscription', saved_files)
    plt.close()
    
    # ========== Plot 12: Instructions Per Task (Normalized) ==========
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(df['oversubscription_ratio'], df['instructions_per_task'] / 1e6, 
            marker='s', linewidth=3, markersize=10, color=colors['instructions'],
            markeredgecolor='black', markeredgewidth=1.5, zorder=3)
    apply_plot_styling(ax, 'Oversubscription Ratio (fibers/workers)', 
                      'Instructions per Task (millions)', 
                      'Instructions per Task vs Oversubscription',
                      colors['instructions'], True, df)
    plt.tight_layout()
    save_figure('12_instructions_per_task_vs_oversubscription', saved_files)
    plt.close()
    
    # ========== Plot 13: Page Faults ==========
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(df['oversubscription_ratio'], df['page_faults'], 
            marker='^', linewidth=3, markersize=10, color=colors['page_faults'],
            markeredgecolor='black', markeredgewidth=1.5, zorder=3)
    apply_plot_styling(ax, 'Oversubscription Ratio (fibers/workers)', 
                      'Page Faults (count)', 
                      'Page Faults vs Oversubscription',
                      colors['page_faults'], False, df)
    plt.tight_layout()
    save_figure('13_page_faults_vs_oversubscription', saved_files)
    plt.close()
    
    # ========== Plot 14: Cache Misses (absolute) ==========
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(df['oversubscription_ratio'], df['cache_misses'] / 1e6, 
            marker='D', linewidth=3, markersize=10, color=colors['cache_miss'],
            markeredgecolor='black', markeredgewidth=1.5, zorder=3)
    apply_plot_styling(ax, 'Oversubscription Ratio (fibers/workers)', 
                      'Cache Misses (millions)', 
                      'Absolute Cache Misses vs Oversubscription',
                      colors['cache_miss'], False, df)
    plt.tight_layout()
    save_figure('14_cache_misses_absolute_vs_oversubscription', saved_files)
    plt.close()
    
    # Print saved files
    print("\n✓ Visualizations saved:")
    for filename in sorted(saved_files):
        print(f"  - {filename}")

def print_summary(df):
    """Print comprehensive summary statistics with normalized metrics."""
    print("\n" + "="*80)
    print("MICROBENCHMARK SUMMARY WITH NORMALIZED PERFORMANCE METRICS")
    print("="*80)
    
    # Find optimal configuration (highest throughput per worker)
    optimal_idx = df['throughput_per_worker'].idxmax()
    optimal = df.loc[optimal_idx]
    
    print(f"\n{'OPTIMAL CONFIGURATION (Peak Per-Worker Efficiency)':^80}")
    print("-" * 80)
    print(f"{'Oversubscription Ratio:':<45} {optimal['oversubscription_ratio']:.1f}x")
    print(f"{'Throughput per Worker:':<45} {optimal['throughput_per_worker']:.2f} tasks/sec/worker")
    print(f"{'Absolute Throughput:':<45} {optimal['throughput']:.2f} tasks/sec")
    print(f"{'System Efficiency:':<45} 100.0% (baseline)")
    print(f"{'Fibers:':<45} {optimal['num_threads']}")
    print(f"{'Workers:':<45} {optimal['num_workers']}")
    print(f"{'IPC:':<45} {optimal['ipc']:.3f}")
    print(f"{'Cache Miss Rate:':<45} {optimal['cache_miss_rate']:.2f}%")
    print(f"{'Branch Miss Rate:':<45} {optimal['branch_miss_rate']:.2f}%")
    print(f"{'Cycles per Task:':<45} {optimal['cycles_per_task']/1e6:.1f}M")
    
    # Performance degradation at maximum oversubscription
    max_oversub_idx = df['oversubscription_ratio'].idxmax()
    max_oversub = df.loc[max_oversub_idx]
    
    efficiency_loss = ((optimal['throughput_per_worker'] - max_oversub['throughput_per_worker']) / 
                       optimal['throughput_per_worker']) * 100
    ipc_loss = ((optimal['ipc'] - max_oversub['ipc']) / optimal['ipc']) * 100
    cache_increase = max_oversub['cache_miss_rate'] - optimal['cache_miss_rate']
    branch_increase = max_oversub['branch_miss_rate'] - optimal['branch_miss_rate']
    cycles_increase = ((max_oversub['cycles_per_task'] - optimal['cycles_per_task']) / 
                      optimal['cycles_per_task']) * 100
    
    print(f"\n{'MAXIMUM OVERSUBSCRIPTION IMPACT':^80}")
    print("-" * 80)
    print(f"{'Oversubscription Ratio:':<45} {max_oversub['oversubscription_ratio']:.1f}x")
    print(f"{'Throughput per Worker:':<45} {max_oversub['throughput_per_worker']:.2f} tasks/sec/worker")
    print(f"{'Per-Worker Efficiency Loss:':<45} {efficiency_loss:.1f}%")
    print(f"{'System Efficiency:':<45} {max_oversub['efficiency_ratio']*100:.1f}%")
    print(f"{'IPC:':<45} {max_oversub['ipc']:.3f}")
    print(f"{'IPC Loss vs Optimal:':<45} {ipc_loss:.1f}%")
    print(f"{'Cache Miss Rate:':<45} {max_oversub['cache_miss_rate']:.2f}%")
    print(f"{'Cache Miss Rate Increase:':<45} +{cache_increase:.2f} pp")
    print(f"{'Branch Miss Rate:':<45} {max_oversub['branch_miss_rate']:.2f}%")
    print(f"{'Branch Miss Rate Increase:':<45} +{branch_increase:.2f} pp")
    print(f"{'Cycles per Task:':<45} {max_oversub['cycles_per_task']/1e6:.1f}M")
    print(f"{'Cycles per Task Increase:':<45} +{cycles_increase:.1f}%")
    
    # Key metrics comparison
    print(f"\n{'NORMALIZED MICROARCHITECTURAL IMPACT':^80}")
    print("-" * 80)
    print(f"{'Metric':<35} {'Optimal':<20} {'Max Oversub':<20} {'Change':>10}")
    print("-" * 80)
    
    metrics_to_compare = [
        ('Throughput/Worker (tasks/s/w)', 'throughput_per_worker', '.2f', ''),
        ('System Efficiency (%)', lambda r: r['efficiency_ratio']*100, '.1f', '%'),
        ('IPC', 'ipc', '.3f', ''),
        ('Cycles per Task (M)', lambda r: r['cycles_per_task']/1e6, '.1f', 'M'),
        ('Cache Miss Rate (%)', 'cache_miss_rate', '.2f', '%'),
        ('Branch Miss Rate (%)', 'branch_miss_rate', '.2f', '%'),
        ('Context Switches', 'context_switches', '.0f', ''),
        ('Fiber Switches', 'thread_switches', '.0f', ''),
        ('Page Faults', 'page_faults', '.0f', ''),
    ]
    
    for name, col, fmt, unit in metrics_to_compare:
        if callable(col):
            opt_val = col(optimal)
            max_val = col(max_oversub)
        else:
            opt_val = optimal[col]
            max_val = max_oversub[col]
        
        change = max_val - opt_val
        change_pct = (change / opt_val * 100) if opt_val != 0 else 0
        
        print(f"{name:<35} "
              f"{opt_val:{fmt}}{unit:<5} "
              f"{max_val:{fmt}}{unit:<5} "
              f"{change_pct:+6.1f}%")
    
    print("\n" + "="*80)
    print("DETAILED RESULTS TABLE (Key Normalized Metrics)")
    print("="*80)
    
    # Select key columns for display
    display_cols = ['oversubscription_ratio', 'throughput_per_worker', 'efficiency_ratio',
                   'ipc', 'cache_miss_rate', 'branch_miss_rate', 'thread_switches']
    
    display_df = df[display_cols].copy()
    display_df['efficiency_ratio'] = display_df['efficiency_ratio'] * 100  # Convert to percentage
    display_df = display_df.rename(columns={
        'oversubscription_ratio': 'Oversub',
        'throughput_per_worker': 'Tput/Worker',
        'efficiency_ratio': 'Efficiency%',
        'ipc': 'IPC',
        'cache_miss_rate': 'Cache%',
        'branch_miss_rate': 'Branch%',
        'thread_switches': 'FiberSwitches'
    })
    
    print(display_df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    print("="*80 + "\n")
    
    # Print key insights
    print("KEY INSIGHTS:")
    print("-" * 80)
    print(f"1. Peak per-worker efficiency at {optimal['oversubscription_ratio']:.1f}x oversubscription")
    print(f"2. {efficiency_loss:.1f}% efficiency loss at {max_oversub['oversubscription_ratio']:.1f}x")
    print(f"   ({optimal['throughput_per_worker']:.0f} → {max_oversub['throughput_per_worker']:.0f} tasks/sec/worker)")
    print(f"3. Cache miss rate increases from {optimal['cache_miss_rate']:.1f}% to "
          f"{max_oversub['cache_miss_rate']:.1f}% (+{cache_increase:.1f} pp)")
    print(f"4. IPC degrades from {optimal['ipc']:.3f} to {max_oversub['ipc']:.3f} "
          f"({ipc_loss:.1f}% loss)")
    print(f"5. Cycles per task increase from {optimal['cycles_per_task']/1e6:.1f}M to "
          f"{max_oversub['cycles_per_task']/1e6:.1f}M (+{cycles_increase:.1f}%)")
    print(f"6. Fiber switches increase from {optimal['thread_switches']:,} to "
          f"{max_oversub['thread_switches']:,}")
    print("\nCONCLUSION: Fiber oversubscription causes significant microarchitectural")
    print("pollution (cache/branch predictor), resulting in measurable efficiency loss.")
    print("="*80 + "\n")

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("MARL FIBER OVERSUBSCRIPTION MICROBENCHMARK")
    print("Performance Analysis with Hardware Counters (NORMALIZED METRICS)")
    print("="*80)
    
    # Load results
    print("\nLoading results...")
    df = load_results()
    
    # Print summary
    print_summary(df)
    
    # Create visualizations
    print("Generating visualizations...")
    create_visualizations(df)
    
    print("\n✓ Analysis complete!")
    print("\nGenerated 14 plots showing:")
    print("  - Normalized efficiency metrics (throughput/worker, system efficiency)")
    print("  - Application metrics (switches, timing)")
    print("  - Microarchitectural metrics (IPC, cache/branch behavior)")
    print("  - System metrics (context switches, page faults)")
    print("\nKey plots to examine:")
    print("  • 01_throughput_per_worker_vs_oversubscription.*  (PRIMARY METRIC)")
    print("  • 02_efficiency_vs_oversubscription.*              (SHOWS DEGRADATION)")
    print("  • 08_cache_miss_rate_vs_oversubscription.*         (CACHE POLLUTION)")
    print("  • 07_ipc_vs_oversubscription.*                     (EFFICIENCY LOSS)")

if __name__ == '__main__':
    main()