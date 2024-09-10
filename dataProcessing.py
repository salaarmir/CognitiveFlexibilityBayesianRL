import pandas as pd

# create dataframe for simulation
def create_df():    

            # initialise results for simulation df
            num_new_rows_sequence = [
                ('', 2000)]

            # initialise start index for TrialIndex column
            current_trial_index = 1

            # create an empty df to store the rows
            simulation_df = pd.DataFrame()

            # generate new rows based on the sequence
            for target_rule, num_rows in num_new_rows_sequence:
                new_rows_batch = pd.DataFrame({
                    'TrialIndex': range(current_trial_index, current_trial_index + num_rows),
                    'TargetRule': [target_rule] * num_rows,
                    'Choice': [''] * num_rows,
                    'Reward': [0] * num_rows
                })
                simulation_df = pd.concat([simulation_df, new_rows_batch], ignore_index=True)
                current_trial_index += num_rows

            # initialise empty df values to be filled as simulation progresses
            simulation_df['MAP_GoRight'] = 0.0
            simulation_df['MAP_GoLeft'] = 0.0
            simulation_df['Precision'] = 0

            return simulation_df