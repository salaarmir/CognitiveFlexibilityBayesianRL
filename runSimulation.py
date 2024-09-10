import utils
import dataProcessing
import numpy as np


def run_simulation(sampling_alg, decay_factor, noise_level, adaptive_decay=False, overtraining=False, iterations=50):
    trials_to_learn_list = {'go right 1': [], 'go left 1': [], 'go right 2': [], 'go left 2': []}

    for _ in range(iterations):

        # create dataframe to store results
        simulation_df = dataProcessing.create_df()
        
        successes = {'left': 0, 'right': 0}
        failures = {'left': 0, 'right': 0}

        # initialise rules and current rule
        rules = ['go right 1', 'go left 1', 'go right 2', 'go left 2']
        rule_index = 0
        rule_change_indices = []
        current_rule = rules[rule_index]

        # set threshold for overtraining
        overtraining_threshold = 100
        overtraining_count = 0

        # initialise learning threshold and consecutive correct counter
        consecutive_correct = 0
        learning_threshold = 10

        # intialise last valid trial index
        last_valid_index = 0

        # initialise further variables needed for simulation
        trial_counter = 1
        learnt_rule = False
        target_rule = None
        all_rules_learned = False

        for index, row in simulation_df.iterrows():

            if target_rule != current_rule:
                
                # update rule in simulation df
                simulation_df.at[index, 'TargetRule'] = current_rule
                rule_change_indices.append(index)
                
                # update target rule
                target_rule = simulation_df.at[index, 'TargetRule']
                
                consecutive_correct = 0
                
                trial_counter = 1
                learnt_rule = False

            simulation_df.at[index, 'TargetRule'] = current_rule

            if sampling_alg == 'Bayes_UCB': 
                choice = utils.bayes_ucb(noise_level, decay_factor, successes, failures, index)
            else:
                choice = utils.thompson_sampling(noise_level, decay_factor, successes, failures)

            # update choice and reward values in dataframe
            simulation_df.at[index, 'Choice'] = choice
            reward = 1 if (choice == 'right' and (target_rule == 'go right 1' or target_rule == 'go right 2')) or (choice == 'left' and (target_rule == 'go left 1' or target_rule == 'go left 2')) else 0
            simulation_df.at[index, 'Reward'] = reward

            # update successes and failures based on decision
            if reward == 1:
                successes[choice] += 1
            else:
                failures[choice] += 1

            # check if current rule is learned
            if not learnt_rule:

                consecutive_correct = utils.check_learning(consecutive_correct, choice, target_rule)

                # check if rule has been learned
                if consecutive_correct >= learning_threshold:
                    learnt_rule = True
                    trials_to_learn_list[target_rule].append(trial_counter)
                    #print(f"Agent has learned the '{target_rule}' rule in {trial_counter} trials. Overtraining is {overtraining}")
                    
                    # adjust decay factor if adaptive_decay is enabled
                    if adaptive_decay:
                        decay_factor *= 1.01
                        print(f"Adaptive Decay Factor adjusted to {decay_factor}")

                    # reset overtraining count to simulate additional 100 trials
                    if overtraining:
                        overtraining_count = 0  

            if learnt_rule:
                if overtraining:

                    # handle overtraining logic
                    overtraining_count += 1

                    # Check if overtraining is complete
                    if overtraining_count >= overtraining_threshold:
            
                        # switch to next rule after overtraining
                        rule_index += 1
                        if rule_index < len(rules):
                            current_rule = rules[rule_index]
                            consecutive_correct = 0
                            learnt_rule = False
                        else:
                            #print("All rules learned.")
                            last_valid_index = index
                            all_rules_learned = True
                else:
                    
                    # switch to next rule immediately after learning
                    rule_index += 1
                    if rule_index < len(rules):
                        current_rule = rules[rule_index]
                        consecutive_correct = 0
                        learnt_rule = False
                    else:
                        #print("All rules learned.")
                        last_valid_index = index
                        all_rules_learned = True

            trial_counter += 1

            map_go_right = utils.calculate_map(successes['right'], failures['right'])
            map_go_left = utils.calculate_map(successes['left'], failures['left'])

            simulation_df.at[index, 'MAP_GoRight'] = map_go_right
            simulation_df.at[index, 'MAP_GoLeft'] = map_go_left

            if all_rules_learned:
                    break
        
    # truncate dataframe to remove unnecessary rows
    simulation_df = simulation_df.iloc[:last_valid_index + 1]
    rule_change_indices.append(len(simulation_df) - 1)

    # calculate cum sums for right and left choices
    simulation_df['CumulativeRightChoice'] = (simulation_df['Choice'] == 'right').cumsum()
    simulation_df['CumulativeLeftChoice'] = (simulation_df['Choice'] == 'left').cumsum()

    # calculate cum difference between right and left choices
    simulation_df['CumulativeChoiceDifference'] = simulation_df['CumulativeRightChoice'] - simulation_df['CumulativeLeftChoice']

    # calculate cum rewards
    simulation_df['CumulativeReward'] = simulation_df['Reward'].cumsum()

    # export updated df to csv
    output_csv_path = "data/simulation_df.csv"
    simulation_df.to_csv(output_csv_path, index=False)

    return simulation_df, trials_to_learn_list, rule_change_indices
