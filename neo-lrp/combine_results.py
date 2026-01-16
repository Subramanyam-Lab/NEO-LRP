"""
Combine individual Excel files from array jobs into single Excel.

Usage:
    python combine_results.py --dataset S_schneider --normalization cost_over_fi --N 110000 --solver vroom --model_type graph_transformer --instances_file /storage/group/azs7266/default/wzk5140/NEO-LRP/neo-lrp/configs/schneider_instances.txt --delete_temp
"""

import os
import argparse
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


def main():
    parser = argparse.ArgumentParser(description='Combine Excel results from array jobs')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['P_prodhon', 'S_schneider', 'T_tuzun', 'B_barreto'])
    parser.add_argument('--normalization', type=str, required=True,
                        choices=['raw', 'minmax', 'cost_over_fi', 'cost_over_fi_minmax'])
    parser.add_argument('--N', type=int, required=True,
                        choices=[110, 1100, 11000, 110000])
    parser.add_argument('--solver', type=str, required=True,
                        choices=['vroom', 'ortools', 'vrpeasy'])
    parser.add_argument('--model_type', type=str, default='deepsets',
                        choices=['deepsets', 'graph_transformer'],
                        help='Neural network architecture')
    parser.add_argument('--instances_file', type=str, required=True)
    parser.add_argument('--delete_temp', action='store_true')
    
    args = parser.parse_args()
    
    with open(args.instances_file, 'r') as f:
        instance_order = [line.strip() for line in f if line.strip()]
    
    print(f"expected instances: {len(instance_order)}")
    print(f"model type: {args.model_type}")
    
    # Include model_type in directory name
    excel_temp_dir = os.path.join(
        OUTPUT_DIR, args.dataset,
        f"{args.model_type}_{args.solver}_{args.normalization}_{args.N}_excel_temp"
    )
    
    if not os.path.exists(excel_temp_dir):
        raise FileNotFoundError(f"excel temp directory not found: {excel_temp_dir}")
    
    all_dfs = []
    missing = []
    found_files = []
    
    for instance_name in instance_order:
        instance_stem = os.path.splitext(instance_name)[0]
        excel_file = os.path.join(excel_temp_dir, f"{instance_stem}.xlsx")
        
        if os.path.exists(excel_file):
            df = pd.read_excel(excel_file, sheet_name='results')
            all_dfs.append(df)
            found_files.append(excel_file)
            print(f"  loaded: {instance_stem}.xlsx")
        else:
            missing.append(instance_name)
            print(f"  missing: {instance_stem}.xlsx")
    
    print(f"\nfound: {len(found_files)} / {len(instance_order)}")
    
    if missing:
        print(f"warning: {len(missing)} missing!")
    
    if not all_dfs:
        raise ValueError("no excel files found!")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"combined: {len(combined)} rows")
    
    # Include model_type in final filename
    final_excel = os.path.join(
        OUTPUT_DIR, args.dataset,
        f"{args.dataset}_{args.model_type}_{args.N}_{args.normalization}_{args.solver}.xlsx"
    )
    
    with pd.ExcelWriter(final_excel, engine='openpyxl') as writer:
        combined.to_excel(writer, sheet_name='results', index=False)
    
    print(f"\ncombined excel: {final_excel}")
    
    # Summary statistics
    successful = combined[combined['Status'] == 'SUCCESS']
    gaps = successful['Optimization_gap_signed'].dropna()
    if len(gaps) > 0:
        print(f"\ngap: avg={gaps.mean():.2f}%, min={gaps.min():.2f}%, max={gaps.max():.2f}%")
    
    if args.delete_temp:
        print(f"\ndeleting {len(found_files)} temp files...")
        for f in found_files:
            os.remove(f)
        if os.path.exists(excel_temp_dir) and not os.listdir(excel_temp_dir):
            os.rmdir(excel_temp_dir)
        print("done.")


if __name__ == "__main__":
    main()