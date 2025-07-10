import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_scale_statistics(results_path):
    """Plot scale statistics from benchmark results.
    
    Args:
        results_path: Path to the CSV file containing benchmark results
    """
    # Read the results
    df = pd.read_csv(results_path)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Scale factors over items
    ax1.plot(df['item_id'], df['optimal_scale'], 'b.', alpha=0.5, label='Scale factors')
    ax1.axhline(y=df['optimal_scale'].mean(), color='r', linestyle='--', label=f'Mean: {df["optimal_scale"].mean():.2f}')
    ax1.set_xlabel('Item ID')
    ax1.set_ylabel('Optimal Scale')
    ax1.set_title('Scale Factors Distribution')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Histogram of scale factors
    ax2.hist(df['optimal_scale'], bins=30, alpha=0.7, color='b')
    ax2.axvline(x=df['optimal_scale'].mean(), color='r', linestyle='--', 
                label=f'Mean: {df["optimal_scale"].mean():.2f}')
    ax2.axvline(x=df['optimal_scale'].median(), color='g', linestyle='--', 
                label=f'Median: {df["optimal_scale"].median():.2f}')
    ax2.set_xlabel('Scale Factor')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Histogram of Scale Factors')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(results_path), 'scale_statistics.png')
    plt.savefig(output_path)
    plt.close()
    
    # Print statistics
    print("\nScale Factor Statistics:")
    print(f"Mean: {df['optimal_scale'].mean():.3f}")
    print(f"Median: {df['optimal_scale'].median():.3f}")
    print(f"Std: {df['optimal_scale'].std():.3f}")
    print(f"Min: {df['optimal_scale'].min():.3f}")
    print(f"Max: {df['optimal_scale'].max():.3f}")
    print(f"25th percentile: {df['optimal_scale'].quantile(0.25):.3f}")
    print(f"75th percentile: {df['optimal_scale'].quantile(0.75):.3f}")

def main():
    parser = argparse.ArgumentParser(description="Analyze scale factors from DUSt3R benchmark results.")
    parser.add_argument('--results', type=str, required=True, help='Path to the benchmark results CSV file.')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results):
        print(f"Results file not found at {args.results}")
        return
        
    plot_scale_statistics(args.results)

if __name__ == "__main__":
    main()
