import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_chart():
    csv_path = '.jules/benchmarks.csv'
    chart_path = '.jules/benchmark_trends.png'

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("Error: DataFrame is empty.")
        return

    # We want to plot:
    # X-axis: Date or CommitHash. Let's use an index or Date.
    df['Date'] = pd.to_datetime(df['Date'])

    # Extract metrics for plotting
    tsfast_ms = df[df['Benchmark_Name'] == 'tsfast_ms_per_window']['Metric_Value'].values
    tsfresh_ms = df[df['Benchmark_Name'] == 'tsfresh_ms_per_window']['Metric_Value'].values
    tsfel_ms = df[df['Benchmark_Name'] == 'tsfel_ms_per_window']['Metric_Value'].values

    tsfast_feats = df[df['Benchmark_Name'] == 'tsfast_compatible_features']['Metric_Value'].values
    tsfresh_feats = df[df['Benchmark_Name'] == 'tsfresh_compatible_features']['Metric_Value'].values
    tsfel_feats = df[df['Benchmark_Name'] == 'tsfel_compatible_features']['Metric_Value'].values

    # In case there are multiple entries over time, we align them by runs
    runs = range(1, len(tsfast_ms) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_fast = 'tab:blue'
    color_fresh = 'tab:green'
    color_fel = 'tab:red'

    # Left Y-axis: ms per window
    ax1.set_xlabel('Benchmark Run')
    ax1.set_ylabel('ms per window with all features', color='black')

    # Plot ms per window (solid lines)
    if len(tsfast_ms) > 0:
        ax1.plot(runs, tsfast_ms, color=color_fast, linestyle='-', label='tsfast ms/window', marker='o')
    if len(tsfresh_ms) > 0:
        ax1.plot(runs, tsfresh_ms, color=color_fresh, linestyle='-', label='tsfresh ms/window', marker='s')
    if len(tsfel_ms) > 0:
        ax1.plot(runs, tsfel_ms, color=color_fel, linestyle='-', label='tsfel ms/window', marker='^')

    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_yscale('log') # often helpful for time series benchmarks
    ax1.legend(loc='upper left')

    # Right Y-axis: compatible features
    ax2 = ax1.twinx()
    ax2.set_ylabel('total compatible features', color='black')

    # Plot compatible features (dashed lines)
    if len(tsfast_feats) > 0:
        ax2.plot(runs, tsfast_feats, color=color_fast, linestyle='--', label='tsfast features', marker='o', alpha=0.6)
    if len(tsfresh_feats) > 0:
        ax2.plot(runs, tsfresh_feats, color=color_fresh, linestyle='--', label='tsfresh features', marker='s', alpha=0.6)
    if len(tsfel_feats) > 0:
        ax2.plot(runs, tsfel_feats, color=color_fel, linestyle='--', label='tsfel features', marker='^', alpha=0.6)

    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(0, max(max(tsfast_feats, default=1), max(tsfresh_feats, default=1), max(tsfel_feats, default=1)) + 10)
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.title("Benchmark Trends: Execution Time vs Feature Support")
    plt.savefig(chart_path)
    print(f"Chart saved to {chart_path}")

if __name__ == "__main__":
    generate_chart()
