import pandas as pd
from scipy.stats import pearsonr, spearmanr
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Calculate correlation between Human and LLM Judge scores.")
    parser.add_argument("--human", default=r"e:\OneDrive\DevelopmentRAG\adapter_training\data\human_scores.csv", help="Path to human scores CSV")
    parser.add_argument("--llm", default=r"e:\OneDrive\DevelopmentRAG\adapter_training\data\data_pipeline_score.csv", help="Path to LLM scores CSV")
    args = parser.parse_args()

    try:
        df_human = pd.read_csv(args.human)
        df_llm = pd.read_csv(args.llm)
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        print("Please ensure both CSV files exist.")
        sys.exit(1)

    # Merge on the 'id' column
    df = pd.merge(df_human, df_llm, on="id", suffixes=('_human', '_llm'))
    if len(df) == 0:
        print("Error: No matching IDs found between the two CSV files.")
        sys.exit(1)

    print("\n" + "="*60)
    print("📊 LLM-as-a-Judge Correlation Analysis")
    print(f"Samples evaluated: {len(df)}")
    print("="*60)

    dims = ["signal_detection", "value_assignment", "completeness"]
    all_human = []
    all_llm = []

    for dim in dims:
        col_h = f"{dim}_human"
        col_l = f"{dim}_llm"
        
        if col_h not in df.columns or col_l not in df.columns:
            print(f"Warning: Columns for {dim} not found. Skipping.")
            continue

        # Drop any rows where one of the scores is missing
        valid = df.dropna(subset=[col_h, col_l])
        
        h_vals = valid[col_h].values
        l_vals = valid[col_l].values
        
        all_human.extend(h_vals)
        all_llm.extend(l_vals)
        
        if len(h_vals) < 2:
            print(f"\n[{dim.upper().replace('_', ' ')}] - Not enough data")
            continue

        pearson, _ = pearsonr(h_vals, l_vals)
        spearman, _ = spearmanr(h_vals, l_vals)
        
        print(f"\n[{dim.upper().replace('_', ' ')}]")
        print(f"  Pearson (Linear):      {pearson:.3f}")
        print(f"  Spearman (Rank):       {spearman:.3f}")

    # Overall correlation across all dimensions
    if len(all_human) >= 2:
        pearson_all, _ = pearsonr(all_human, all_llm)
        spearman_all, _ = spearmanr(all_human, all_llm)
        
        print("\n" + "-"*60)
        print("[OVERALL ACROSS ALL DIMENSIONS]")
        print(f"  Pearson Correlation:   {pearson_all:.3f}")
        print(f"  Spearman Correlation:  {spearman_all:.3f}")
        
        print("\nInterpretation Guide:")
        print("  > 0.8: Excellent agreement (Ready for production)")
        print("  > 0.6: Good agreement (Acceptable but could refine prompt)")
        print("  < 0.5: Poor agreement (Judge is misaligned with human criteria)")
        print("="*60 + "\n")

if __name__ == "__main__":
    main()
