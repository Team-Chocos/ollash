import pandas as pd
import json
import glob
import os
from pathlib import Path


def format_leaderboard_display(leaderboard: pd.DataFrame) -> str:
    """Format leaderboard for better terminal display"""
    # Create a formatted string representation
    header = "┌─────┬─────────────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┐"
    separator = "├─────┼─────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┤"
    footer = "└─────┴─────────────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┘"
    
    # Column headers
    col_header = "│ Rank│ Model                   │ Composite Score  │ Sequence Sim     │ Jaccard Sim      │ Structure Sim    │ Exact Match Rate │ Functional Equiv │ Success Rate     │ Total Samples    │"
    
    formatted_output = f"{header}\n{col_header}\n{separator}\n"
    
    for i, row in leaderboard.iterrows():
        rank = str(i).center(4)
        model = row['Model'][:23].ljust(23)
        comp_score = f"{row['Composite Score']:.3f}".center(16)
        seq_sim = f"{row['Sequence Similarity']:.3f}".center(16)
        jaccard_sim = f"{row['Jaccard Similarity']:.3f}".center(16)
        struct_sim = f"{row['Structure Similarity']:.3f}".center(16)
        exact_match = f"{row['Exact Match Rate']:.3f}".center(16)
        func_equiv = f"{row['Functional Equiv Rate']:.3f}".center(16)
        success_rate = f"{row['Success Rate']:.3f}".center(16)
        total_samples = str(int(row['Total Samples'])).center(16)
        
        row_line = f"│ {rank}│ {model} │ {comp_score} │ {seq_sim} │ {jaccard_sim} │ {struct_sim} │ {exact_match} │ {func_equiv} │ {success_rate} │ {total_samples} │"
        formatted_output += f"{row_line}\n"
    
    formatted_output += footer
    return formatted_output


def print_banner():
    """Print a nice banner for the results display"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                   NL2SH BENCHMARK RESULTS                                                                                           ║
║                                                              Natural Language to Shell Command Evaluation                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_section_header(title: str):
    """Print a formatted section header"""
    width = 80
    print(f"\n{'═' * width}")
    print(f"║ {title.center(width-4)} ║")
    print(f"{'═' * width}")


def print_example_predictions(predictions: list, model_name: str):
    """Print formatted example predictions"""
    print(f"\n┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐")
    print(f"│ Example Predictions from Top Model: {model_name.ljust(60)} │")
    print(f"└─────────────────────────────────────────────────────────────────────────────────────────────────────┘")
    
    for i, pred in enumerate(predictions[:3], 1):
        print(f"\n┌── Query {i} " + "─" * 70)
        print(f"│ Input:        {pred['nl']}")
        print(f"│ Expected:     {pred['expected']}")
        if pred.get('expected_alt'):
            print(f"│ Expected Alt: {pred['expected_alt']}")
        print(f"│ Predicted:    {pred['predicted']}")
        print(f"│ Score:        {pred['composite_score']:.3f}")
        print(f"└" + "─" * 77)


def display_existing_results():
    print_banner()
    
    # Check for final results first in the new directory structure
    benchmark_dir = Path.home() / '.ollash' / 'benchmarks'
    final_results_path = benchmark_dir / 'final_results.json'
    
    print_section_header("LOADING BENCHMARK RESULTS")
    
    if final_results_path.exists():
        print(f"Loading final results from: {final_results_path}")
        with open(final_results_path, 'r') as f:
            results = json.load(f)
        print("Final results loaded successfully")
    else:
        print("No final results found. Checking for intermediate results...")
        
        # Fall back to intermediate results in the new directory
        result_files = list(benchmark_dir.glob('intermediate_results_*.json'))
        
        if not result_files:
            print("No benchmark results found in ~/.ollash/benchmarks/")
            
            # Final fallback to current directory (old behavior)
            print("Checking current directory for legacy results...")
            result_files = glob.glob('intermediate_results_*.json')
            
            if not result_files:
                print("No benchmark results found!")
                return
            
            # Get the most recent file from current directory
            latest_file = max(result_files, key=os.path.getctime)
            print(f"Loading legacy results from: {latest_file}")
            
            with open(latest_file, 'r') as f:
                results = json.load(f)
        else:
            # Get the most recent file from benchmarks directory
            latest_file = max(result_files, key=lambda p: p.stat().st_ctime)
            print(f"Loading intermediate results from: {latest_file}")
            
            with open(latest_file, 'r') as f:
                results = json.load(f)
    
    print_section_header("BENCHMARK LEADERBOARD")
    
    # Create leaderboard
    leaderboard_data = []
    for result in results:
        leaderboard_data.append({
            'Model': result['model'],
            'Composite Score': result['avg_composite_score'],
            'Sequence Similarity': result['avg_sequence_similarity'],
            'Jaccard Similarity': result['avg_jaccard_similarity'],
            'Structure Similarity': result['avg_structure_similarity'],
            'Exact Match Rate': result['exact_match_rate'],
            'Functional Equiv Rate': result['functional_equivalence_rate'],
            'Success Rate': result['success_rate'],
            'Total Samples': result['total_samples']
        })
    
    df = pd.DataFrame(leaderboard_data)
    df = df.sort_values(['Composite Score', 'Exact Match Rate'], ascending=False)
    df = df.reset_index(drop=True)
    df.index += 1
    
    # Display formatted leaderboard
    print("\nNL2SH BENCHMARK LEADERBOARD")
    print(format_leaderboard_display(df))
    
    # Display summary statistics
    print_section_header("BENCHMARK SUMMARY")
    print(f"Total Models Tested:      {len(df)}")
    print(f"Best Composite Score:     {df['Composite Score'].max():.3f}")
    print(f"Average Composite Score:  {df['Composite Score'].mean():.3f}")
    print(f"Best Exact Match Rate:    {df['Exact Match Rate'].max():.3f}")
    print(f"Average Success Rate:     {df['Success Rate'].mean():.3f}")
    print(f"Total Samples per Model:  {df['Total Samples'].iloc[0]}")
    
    print_section_header("TOP MODEL ANALYSIS")
    
    # Show example predictions from top model
    top_model_results = sorted(results, key=lambda x: x['avg_composite_score'], reverse=True)[0]
    print(f"Top performing model: {top_model_results['model']}")
    print(f"Composite Score: {top_model_results['avg_composite_score']:.3f}")
    print(f"Exact Match Rate: {top_model_results['exact_match_rate']:.3f}")
    print(f"Success Rate: {top_model_results['success_rate']:.3f}")
    
    print_example_predictions(top_model_results['predictions'], top_model_results['model'])
    
    # Performance breakdown by difficulty (if available)
    if 'predictions' in top_model_results and len(top_model_results['predictions']) > 0:
        predictions = top_model_results['predictions']
        
        # Group by difficulty if available
        difficulty_stats = {}
        for pred in predictions:
            # Check if we have difficulty info from the dataset loading
            difficulty = "Unknown"  # Default since we don't store difficulty in predictions
            
            if difficulty not in difficulty_stats:
                difficulty_stats[difficulty] = {'count': 0, 'scores': []}
            
            difficulty_stats[difficulty]['count'] += 1
            difficulty_stats[difficulty]['scores'].append(pred['composite_score'])
        
        if len(difficulty_stats) > 1:  # Only show if we have difficulty breakdown
            print_section_header("PERFORMANCE BY DIFFICULTY")
            for difficulty, stats in difficulty_stats.items():
                avg_score = sum(stats['scores']) / len(stats['scores'])
                print(f"Difficulty {difficulty}: {stats['count']} samples, Avg Score: {avg_score:.3f}")


if __name__ == "__main__":
    display_existing_results()
