# Instructions

This part of the repo implements code related to the **value of location allocation decision ablation study** from our paper. Please refer to the corresponding section in our paper for details on the methodology.

## Step 1: Run Main

Run `main.py` to solve FLP-VRP on benchmark instances of Set P:

```bash
python main.py
```

**Before running**, please verify these paths in `main.py`:
- `directory_path` - Path to benchmark instances (Set P)
- `existing_excel_file` - Path for output Excel file

**Expected outputs:**
- `lp_results.xlsx` - Excel file with results on set P
- `results/solutions_flpvrp/` - JSON solution files for each instance

## Step 2: Generate Comparison Tables

Run `generate_tables.py` to create instance-wise comparisons:

```bash
python generate_tables.py
```

**Before running**, verify these paths in `generate_tables.py`:
- `DIR_FLP` - Path to FLP-VRP solutions excel that was generated in Step 1
- `DIR_NEO` - Path to NEO-DS solutions that was generated when you ran the NEO-DS (main section results of our paper)

**Expected outputs:**
- `surrogate_analysis.csv` - Numerical comparison data
- `table_value_loc_alloc_instances.tex` - Instance-wise LaTeX table (presented in paper)

## Step 3: Visualizations

Run `plot_comparison.py` for instance-level visual comparisons:

```bash
python plot_comparison.py
```

**Before running**:
- Verify `DIR_FLP` and `DIR_NEO` paths in `plot_comparison.py`
- **Requires TeXLive** (or similar LaTeX distribution) for rendering

**Expected outputs:**
- `{instance}_FLPvsNEO_2x2.pdf` - 2×2 panel plots comparing FLP-VRP and NEO-DS solutions (presented in the paper)
