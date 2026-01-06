from deephyper.hpo import HpProblem
from deephyper.evaluator import Evaluator
from deephyper.hpo import CBO
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import os
import argparse

from train import master
from runner import run

num_instances_env = os.getenv('num_instances')
normalization_mode = os.getenv('normalization_mode')
mode = os.getenv('mode')
output_dir = os.getenv('output_dir', '.')

max_evals_env = os.getenv('max_evals')
if max_evals_env is not None:
    max_evals_int = int(max_evals_env)
else:
    raise ValueError("Error: Please provide the 'max_evals' environment variable.")

storage_path_csvs = output_dir

parser = argparse.ArgumentParser()
parser.add_argument('--surrogate_model', type=str, default='GP', help='Type of surrogate model to use')
args = parser.parse_args()

surrogate_model = args.surrogate_model

Problem = HpProblem()
Problem.add_hyperparameter([4, 6, 8], "latent_space_dimension")
Problem.add_hyperparameter([2, 3, 4, 5, 6], "num_layers_phi")
Problem.add_hyperparameter([1], "num_layers_rho")  
Problem.add_hyperparameter([32, 64, 128, 256, 512, 1024], "neurons_per_layer_phi") 
Problem.add_hyperparameter([4, 6, 8], "neurons_per_layer_rho") 
Problem.add_hyperparameter([15, 20], "early_stopping_patience")
Problem.add_hyperparameter([32], "batch_size") 
Problem.add_hyperparameter([0.001], "learning_rate")
Problem.add_hyperparameter([50, 100, 200, 400, 600, 800], "num_epochs")
Problem.add_hyperparameter(['Adam'], "optimizer")  
Problem.add_hyperparameter(['MSE'], "loss_function")  
Problem.add_hyperparameter(['relu'], "activation_function")

if __name__ == "__main__":
    
    multiprocessing.set_start_method('spawn')

    csv_file_name = os.path.join(storage_path_csvs, f'best_config_{surrogate_model}_{mode}_{normalization_mode}.csv')
    
    if os.path.isfile(csv_file_name):
        df = pd.read_csv(csv_file_name)
        best_row = df.loc[df['objective'].idxmin()]        
        initial_points = [best_row.drop('objective').to_dict()]
    else:
        initial_points = []
    
    evaluator = Evaluator.create(run, method="process", method_kwargs={"num_workers": 4})

    search = CBO(problem=Problem, evaluator=evaluator, random_state=42, surrogate_model=surrogate_model, initial_points=initial_points)

    results = search.search(max_evals=max_evals_int)
    
    results['objective'] = pd.to_numeric(results['objective'], errors='coerce')
    results = results.dropna(subset=['objective'])

    i_max = results.objective.argmax()
    
    best_config = results.iloc[i_max][:-3].to_dict()
    best_config = {key.replace('p:', ''): value for key, value in best_config.items()}

    print("Best Configuration:")
    print(best_config)

    results['best_objective'] = -results['objective'].cummax()
    best_config['objective'] = results['best_objective'].iloc[-1]

    plt.figure(figsize=(10, 5))
    plt.plot(results.index, results['best_objective'])
    plt.xlabel('Evaluation')
    plt.ylabel('Best Avg Test Loss (Cost)')
    plot_file_name = os.path.join(storage_path_csvs, f'BestObjvsE_{mode}_{normalization_mode}.png')
    # plt.savefig(plot_file_name)
    plt.close()

    metrics = master(best_config, metrics=True, exportonnx=True, testing=True, seed=42, N_dim=3, 
                     checkpoint_suffix=f"final_best_{mode}_{num_instances_env}")

    desired_metrics = ['test_cost_loss', 'test_opt_gap_cost_percent', 'test_opt_gap_cost_abs_percent', 'avg_new_metrics_abs']
    for metric in desired_metrics:
        if metric in metrics:
            value = metrics[metric].item() if hasattr(metrics[metric], 'item') else metrics[metric]
            best_config[metric] = value

    best_config_df = pd.DataFrame([best_config])

    if not os.path.isfile(csv_file_name):
        best_config_df.to_csv(csv_file_name, index=False)
    else:
        best_config_df.to_csv(csv_file_name, mode='a', header=False, index=False)

    for key, value in metrics.items():
        print(f'{key}: {value}\n')