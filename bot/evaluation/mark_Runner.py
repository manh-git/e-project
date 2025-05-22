def save_results(df, base_path="/content/drive/MyDrive/game_ai"):
    os.makedirs(base_path, exist_ok=True)
    if df.empty:
        return None, None

    csv_path = f"{base_path}/benchmark_results.csv"
    df.to_csv(csv_path, index=False)

    # Create a directory for individual plots
    plots_dir = os.path.join(base_path, "individual_plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Get unique algorithms
    algorithms = df['algorithm'].unique()
    
    # Create individual plots for each algorithm
    plot_paths = []
    for algo in algorithms:
        algo_df = df[df['algorithm'] == algo]
        
        plt.figure(figsize=(10, 6))
        plt.plot(algo_df['run'], algo_df['score'], marker='o', color='blue')
        
        plt.title(f"Performance of {algo}", fontsize=16)
        plt.xlabel("Run Number", fontsize=14)
        plt.ylabel("Score", fontsize=14)
        plt.grid(True)
        
        # Save individual plot
        plot_path = os.path.join(plots_dir, f"{algo.replace(' ', '_')}_plot.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        plot_paths.append(plot_path)

    # Create a combined plot with legend on the right
    plt.figure(figsize=(14, 8))
    
    # Adjust the right margin to make space for the legend
    plt.subplots_adjust(right=0.75)
    
    for algo in algorithms:
        algo_df = df[df['algorithm'] == algo]
        plt.plot(algo_df['run'], algo_df['score'], marker='o', label=algo)
    
    plt.title("Algorithm Comparison", fontsize=16)
    plt.xlabel("Number of Runs", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.grid(True)
    
    # Place legend on the right side outside the plot area
    plt.legend(title="Algorithms", fontsize=12, 
               bbox_to_anchor=(1.05, 1), loc='upper left')
    
    combined_plot_path = f"{base_path}/combined_plot.png"
    plt.savefig(combined_plot_path, bbox_inches='tight')
    plt.close()
    
    return csv_path, plot_paths, combined_plot_path
