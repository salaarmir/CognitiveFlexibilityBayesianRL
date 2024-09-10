import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta
import seaborn as sns
import statsmodels.api as sm

import createPlots
import runSimulation

# parameter grid to see effect of different values of decay_factor and noise_level
decay_factors = [0.5, 0.7, 0.9]
noise_levels = [0.1, 0.2, 0.3]

# specify sampling algorithm
sample_alg = 'Thompson'
sampling_algorithms = ['Thompson', 'Bayes-UCB']

# set if agents should be overtrained or not 
conditions = [True, False]

# intialise list of empty dataframes
simulation_dfs_dict = {'overtrained' : [[None for _ in range(len(noise_levels))] for _ in range(len(decay_factors))], 'normal' : [[None for _ in range(len(noise_levels))] for _ in range(len(decay_factors))]}

# store results
results = []

unfiltered_data = pd.DataFrame(columns=['Decay Factor', 'Noise Level', 'Sampling Algorithm', 'Rule', 'Trials to Learn', 'Overtrained'])

unfiltered_rows_list = []

# iterate through overtraining vs normal conditions
for isOvertraining in conditions:

    condition_key = 'overtrained' if isOvertraining else 'normal'

    # iterate through decay_factors and noise_levels to run simulations for different parameters
    for i, decay in enumerate(decay_factors):
        for j, noise_level in enumerate(noise_levels):
            print(f"Running simulation for decay_factor={decay}, noise_level={noise_level}, overtraining={isOvertraining}, sampling algorithm={sample_alg}")

            simulation_df, trials_to_learn_list, rule_change_indices = runSimulation.run_simulation(sample_alg, decay, noise_level, overtraining=isOvertraining)
            
            # append unfiltered results to dataframe
            for rule, trials in trials_to_learn_list.items():
                for trial in trials:  
                    
                    unfiltered_rows_list.append({
                        'Decay Factor': decay,
                        'Noise Level': noise_level,
                        'Sampling Algorithm': sample_alg,
                        'Rule': rule,
                        'Trials to Learn': trial,
                        'Overtrained': isOvertraining,
                        })

            avg_trials_to_learn = {rule: np.mean(trials) if trials else None for rule, trials in trials_to_learn_list.items()}    
            results.append((decay, noise_level, sample_alg, avg_trials_to_learn, isOvertraining))
            
            simulation_dfs_dict[condition_key][i][j] = simulation_df

unfiltered_data = pd.DataFrame(unfiltered_rows_list)

# convert results to df for easier analysis
results_df = pd.DataFrame(results, columns=['Decay Factor', 'Noise Level', 'Sampling Algorithm', 'Trials to Learn', 'Overtrained'])

trials_to_learn_dict = results_df['Trials to Learn']

# convert learning trials dictionary to float values
trials_to_learn_df = pd.json_normalize(trials_to_learn_dict)
results_df = results_df.drop(columns=['Trials to Learn']) 
results_df = pd.concat([results_df, trials_to_learn_df], axis=1)

# export unfiltered df to csv
output_csv_path = "BayesianRLCode/data/unfiltered_results.csv"

unfiltered_data.to_csv(output_csv_path, index=False)

# export results df to csv
output_csv_path = "BayesianRLCode/data/filtered_results.csv"
results_df.to_csv(output_csv_path, index=False)

# create analysis plots
createPlots.conductANOVA(unfiltered_data)
createPlots.plotLearningCurves(decay_factors, noise_levels, sample_alg, simulation_dfs_dict)
createPlots.plotTotalCorrect(decay_factors, noise_levels, sample_alg, simulation_dfs_dict['overtrained'])
createPlots.plotCumulativeDistributions(decay_factors, noise_levels, sample_alg, simulation_dfs_dict['overtrined'])
createPlots.plotMAPProbabilities(decay_factors, noise_levels, sample_alg, simulation_dfs_dict['normal'])
createPlots.plotStatistics(unfiltered_data)