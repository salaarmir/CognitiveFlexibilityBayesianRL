import numpy as np
from scipy.stats import beta
import pandas as pd

# define thomposon sampling function 
def thompson_sampling(noise_level, decay_factor, successes, failures):

    # sample from both beta distributions (left and right lever)
    beta_left = np.random.beta(successes['left'] * decay_factor + 1, failures['left'] * decay_factor + 1) + np.random.normal(0, noise_level)
    beta_right = np.random.beta(successes['right'] * decay_factor + 1, failures['right'] * decay_factor + 1) + np.random.normal(0, noise_level)

    # pick larger prob value from sampled distributions
    if beta_right > beta_left:
        return 'right'
    else:
        return 'left'

def bayes_ucb(noise_level, decay_factor, successes, failures, t):

    # calculate ucb for both arms using the inverse cdf (quantile function) of the beta distrib
    ucb_left = beta.ppf(1 - 1 / (t + 1), successes['left'] * decay_factor + 1, failures['left'] * decay_factor + 1) + np.random.normal(0, noise_level)
    ucb_right = beta.ppf(1 - 1 / (t + 1), successes['right'] * decay_factor + 1, failures['right'] * decay_factor + 1) + np.random.normal(0, noise_level)

    # pick arm with higher ucb value 
    if ucb_right > ucb_left:
        return'right' 
    else: 
        return'left'
    
# check whether agent has correctly learnt the current rule
def check_learning(consecutive_correct, choice, target_rule):
    if (choice == 'right' and (target_rule == 'go right 1' or target_rule == 'go right 2')) or (choice == 'left' and (target_rule == 'go left 1' or target_rule == 'go left 2')):
        consecutive_correct += 1
    else:
        consecutive_correct = 0
    return consecutive_correct

# calculate map probablities for simulation
def calculate_map(successes, failures):
  
    # calculate map of beta distribution using prob density function
    x = np.arange(0, 1, 0.001)
    y = beta.pdf(x, successes, failures)
    map_prob = x[np.argmax(y)]
    return map_prob

# calculate total correct for left and right rules
def calculate_totalcorrect(simulation_df):

    right_total = 0
    left_total = 0

    right_correct = 0
    left_correct = 0
    for index, row in simulation_df.iterrows():

        if row['TargetRule'] == 'go right 1' or row['TargetRule'] == 'go right 2':
            right_total += 1
            if row['Choice'] == 'right':
                right_correct += 1
        else:
            left_total += 1
            if row['Choice'] == 'left':
                left_correct += 1

    right_precision = right_correct / right_total if right_total > 0 else 0
    left_precision = left_correct / left_total if left_total > 0 else 0

    return right_precision, left_precision

# get indices when rules were reversed during simulation
def getRuleChangeIndices(simulation_df):

    current_rule = simulation_df.at[2, 'TargetRule'] 
    rule_change_indices = []
    rule_change_index = 0
    for index, row in simulation_df.iterrows():
        
        rule_change_index += 1
        if simulation_df.at[index, 'TargetRule'] != current_rule:
            rule_change_indices.append(rule_change_index)
            current_rule = simulation_df.at[index, 'TargetRule']

    rule_change_indices.append(len(simulation_df) - 1)

    return rule_change_indices


