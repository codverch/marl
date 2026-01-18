#!/usr/bin/env python3
"""
Visualization script for MARL fiber oversubscription microbenchmark results.
Generates plots showing the performance impact of fiber oversubscription.
Updated with professional aesthetics.
"""

import sys
import os
import subprocess

# Install required packages
def install_packages():
    """Install required packages if not already installed."""
    packages = ['matplotlib', 'pandas']
    
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

def load_results(filename='benchmark_results.csv'):
    """Load benchmark results from CSV file."""
    if not os.path.exists(filename):
        print(f"Error: {filename} not found!")
        print("Run the microbenchmark first to generate results.")
        sys.exit(1)
    
    return pd.read_csv(filename)

def create_visualizations(df):
    """Create comprehensive visualization plots as separate figures."""
    
    # Set up professional plotting style
    plt.rcParams.update({
        'font.size': 14, 
        'font.family': 'serif'
    })
    
    # Define professional colors
    color_palette = {
        'throughput': '#CC0066',      # magenta
        'switches': '#006666',         # dark teal
        'computation': '#F18F01',      # orange
        'total_time': 'lime',          # lime
        'optimal': '#060771'           # maroon
    }
    
    # Find optimal point for reference
    optimal_idx = df['oversubscription_ratio'].sub(1.0).abs().idxmin()
    
    saved_files = []
    
    # ========== Plot 1: Throughput vs Oversubscription Ratio ==========
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(df['oversubscription_ratio'], df['throughput'], 
             marker='o', linewidth=3, markersize=10, color=color_palette['throughput'],
             markeredgecolor='black', markeredgewidth=1.5, zorder=3)
    ax1.set_xlabel('Oversubscription Ratio (fibers/workers)', fontsize=20)
    ax1.set_ylabel('Throughput (tasks/sec)', fontsize=20)
    ax1.set_title('Throughput vs Fiber Oversubscription', fontsize=24, fontweight='bold', pad=15)
    ax1.grid(True, axis='y', alpha=0.8, linestyle=':', color='black', linewidth=2.0, zorder=0)
    ax1.tick_params(axis='both', labelsize=16)
    
    # Highlight optimal point
    ax1.axvline(x=df.loc[optimal_idx, 'oversubscription_ratio'], 
                color=color_palette['optimal'], linestyle='--', alpha=0.8, 
                linewidth=2.5, label='Optimal (~1x)', zorder=2)
    legend1 = ax1.legend(frameon=True, fancybox=False, shadow=False,
                        loc='best', fontsize=18, edgecolor='black')
    legend1.get_frame().set_linewidth(2.0)
    legend1.get_frame().set_facecolor('white')
    legend1.get_frame().set_alpha(0.95)
    
    # Add borders
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(2.5)
    
    plt.tight_layout()
    for ext in ['png', 'pdf', 'eps']:
        filename = f'throughput_vs_oversubscription.{ext}'
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        saved_files.append(filename)
    plt.close()
    
    # ========== Plot 2: Fiber Switches vs Oversubscription ==========
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    ax2.plot(df['oversubscription_ratio'], df['thread_switches'], 
             marker='s', linewidth=3, markersize=10, color=color_palette['switches'],
             markeredgecolor='black', markeredgewidth=1.5, zorder=3)
    ax2.set_xlabel('Oversubscription Ratio (fibers/workers)', fontsize=20)
    ax2.set_ylabel('Fiber Switches (count)', fontsize=20)
    ax2.set_title('Fiber Switches vs Oversubscription', fontsize=24, fontweight='bold', pad=15)
    ax2.grid(True, axis='y', alpha=0.8, linestyle=':', color='black', linewidth=2.0, zorder=0)
    ax2.tick_params(axis='both', labelsize=16)
    
    # Add borders
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(2.5)
    
    plt.tight_layout()
    for ext in ['png', 'pdf', 'eps']:
        filename = f'fiber_switches_vs_oversubscription.{ext}'
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        saved_files.append(filename)
    plt.close()
    
    # ========== Plot 3: Average Computation Time vs Oversubscription ==========
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    ax3.plot(df['oversubscription_ratio'], df['avg_computation_time_ms'], 
             marker='^', linewidth=3, markersize=10, color=color_palette['computation'],
             markeredgecolor='black', markeredgewidth=1.5, zorder=3)
    ax3.set_xlabel('Oversubscription Ratio (fibers/workers)', fontsize=20)
    ax3.set_ylabel('Avg Computation Time (ms)', fontsize=20)
    ax3.set_title('Computation Time Increase with Oversubscription', 
                  fontsize=24, fontweight='bold', pad=15)
    ax3.grid(True, axis='y', alpha=0.8, linestyle=':', color='black', linewidth=2.0, zorder=0)
    ax3.tick_params(axis='both', labelsize=16)
    
    # Add borders
    for spine in ax3.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(2.5)
    
    plt.tight_layout()
    for ext in ['png', 'pdf', 'eps']:
        filename = f'computation_time_vs_oversubscription.{ext}'
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        saved_files.append(filename)
    plt.close()
    
    # ========== Plot 4: Total Execution Time ==========
    fig4, ax4 = plt.subplots(figsize=(10, 7))
    ax4.plot(df['oversubscription_ratio'], df['total_time_ms'], 
             marker='D', linewidth=3, markersize=10, color=color_palette['total_time'],
             markeredgecolor='black', markeredgewidth=1.5, zorder=3)
    ax4.set_xlabel('Oversubscription Ratio (fibers/workers)', fontsize=20)
    ax4.set_ylabel('Total Time (ms)', fontsize=20)
    ax4.set_title('Total Execution Time vs Oversubscription', 
                  fontsize=24, fontweight='bold', pad=15)
    ax4.grid(True, axis='y', alpha=0.8, linestyle=':', color='black', linewidth=2.0, zorder=0)
    ax4.tick_params(axis='both', labelsize=16)
    
    # Add borders
    for spine in ax4.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(2.5)
    
    plt.tight_layout()
    for ext in ['png', 'pdf', 'eps']:
        filename = f'total_time_vs_oversubscription.{ext}'
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        saved_files.append(filename)
    plt.close()
    
    # Print saved files
    print("\n✓ Visualizations saved to:")
    for filename in saved_files:
        print(f"  - {filename}")

def print_summary(df):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("MICROBENCHMARK SUMMARY")
    print("="*70)
    
    # Find optimal configuration
    optimal_idx = df['throughput'].idxmax()
    optimal = df.loc[optimal_idx]
    
    print(f"\n{'Optimal Configuration:':<30} {optimal['oversubscription_ratio']:.1f}x oversubscription")
    print(f"{'Peak Throughput:':<30} {optimal['throughput']:.2f} tasks/sec")
    print(f"{'Fibers:':<30} {optimal['num_threads']}")
    print(f"{'Workers:':<30} {optimal['num_workers']}")
    
    # Performance degradation at high oversubscription
    worst_idx = df['throughput'].idxmin()
    worst = df.loc[worst_idx]
    
    degradation = ((optimal['throughput'] - worst['throughput']) / optimal['throughput']) * 100
    
    print(f"\n{'Worst Configuration:':<30} {worst['oversubscription_ratio']:.1f}x oversubscription")
    print(f"{'Throughput:':<30} {worst['throughput']:.2f} tasks/sec")
    print(f"{'Performance Loss:':<30} {degradation:.1f}%")
    
    print("\n" + "="*70)
    print("\nDetailed Results:")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70 + "\n")

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("MARL Fiber Oversubscription Microbenchmark - Results Visualization")
    print("="*70)
    
    # Load results
    df = load_results()
    
    # Print summary
    print_summary(df)
    
    # Create visualizations
    print("Generating visualizations...")
    create_visualizations(df)
    
    print("\n✓ Analysis complete!")

if __name__ == '__main__':
    main()