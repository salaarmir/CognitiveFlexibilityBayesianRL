import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta
import utils
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

def plotLearningCurves(decay_factors, noise_levels, sampling_alg, simulation_dfs):
  
    fig, axs = plt.subplots(2, 1, figsize=(14, 10)) 
    index = 0

    for training in simulation_dfs.keys():
                
        # initialise variable to store shortest simulation time
        min_length = float('inf')

        # initalise dict for fastest agents info
        fastest_agent_info = {}

        for i, decay in enumerate(decay_factors):
            for j, noise_level in enumerate(noise_levels):
                
                # plot cumulative rewards 
                axs[index].plot(simulation_dfs[training][i][j]['TrialIndex'], simulation_dfs[training][i][j]['CumulativeReward'], label=f"Decay: {decay}, Noise: {noise_level} Sampling Algorithm: {sampling_alg}")
                
                if len(simulation_dfs[training][i][j]['TrialIndex']) < min_length:
                    min_length = len(simulation_dfs[training][i][j]['TrialIndex'])
                    
                    # store info of fastest agents
                    fastest_agent_info = {
                        'Decay Factor': decay,
                        'Noise Level': noise_level,
                        'Training Type': training,
                        'Sampling Algorithm': sampling_alg,
                        'Trial Length': len(simulation_dfs[training][i][j]['TrialIndex'])
                    }

                # set title based on training
                if training == 'overtrained':
                    axs[index].set_title('Cumulative Reward Over Time For Over-trained Agents')
                else:
                    axs[index].set_title('Cumulative Reward Over Time For Normally Trained Agents')
        
        # print fastest agents info
        print("Fastest agent details:")
        print(f"Training Type: {fastest_agent_info['Training Type']}")
        print(f"Decay Factor: {fastest_agent_info['Decay Factor']}")
        print(f"Noise Level: {fastest_agent_info['Noise Level']}")
        print(f"Sampling Algorithm: {fastest_agent_info['Sampling Algorithm']}")
        print(f"Trial Length: {fastest_agent_info['Trial Length']}")
        index += 1


    for ax in axs.flat:
        ax.set(xlabel='Trial Index', ylabel='Cumulative Reward')
        plt.legend()

    plt.savefig("images/learning_curves.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

def plotCumulativeDistributions(decay_factors, noise_levels, sampling_alg, simulation_dfs):

    fig, axs = plt.subplots(len(decay_factors), len(noise_levels), figsize=(18,12))

    for i, decay in enumerate(decay_factors):
        for j, noise_level in enumerate(noise_levels):

            # determine rule change points
            rule_change_indices = utils.getRuleChangeIndices(simulation_dfs[i][j])

            if len(rule_change_indices) >= 4:
                first_right_end = rule_change_indices[0]
                first_left_start = rule_change_indices[0] + 1
                first_left_end = rule_change_indices[1]
                second_right_start = rule_change_indices[1] + 1
                second_right_end = rule_change_indices[2]
                second_left_start = rule_change_indices[2] + 1
                second_left_end = rule_change_indices[3]
            else:
                # handle exception
                print("Not enough rule changes detected.")

            # plot cum choice distributions
            axs[i][j].plot(simulation_dfs[i][j]['TrialIndex'], simulation_dfs[i][j]['CumulativeChoiceDifference'], label='Choice Difference (Right - Left)', color='black') # plot cum choice difference

            # plot cum rewards
            axs[i][j].plot(simulation_dfs[i][j]['TrialIndex'], simulation_dfs[i][j]['CumulativeReward'], label='Cumulative Reward', color='grey')

            # add vertical lines to indicate rule changes
            axs[i][j].axvline(x=first_right_end, color='grey', linestyle='--', lw=1)
            axs[i][j].axvline(x=first_left_end, color='grey', linestyle='--', lw=1)
            axs[i][j].axvline(x=second_right_end, color='grey', linestyle='--', lw=1)

            axs[i][j].axvspan(0, first_right_end, color='lightblue', alpha=0.3, label='go right 1')
            axs[i][j].axvspan(first_right_end, first_left_end, color='lightgreen', alpha=0.3, label='go left 1')
            axs[i][j].axvspan(first_left_end, second_right_end, color='lightcoral', alpha=0.3, label='go right 2')
            axs[i][j].axvspan(second_right_end, second_left_end, color='lightyellow', alpha=0.3, label='go left 2')
   
            # add plot details
            axs[i][j].set_xlabel('Trial Index')
            axs[i][j].set_ylabel('Cumulative Distribution')
            axs[i][j].set_title(f'Cumulative Distribution for Over-trained Agents ($\\gamma={decay}$ , $\\eta={noise_level}$), Sampling Algorithm={sampling_alg}', fontsize=12)
            axs[i][j].legend(loc='upper right')
            axs[i][j].grid(True)

    plt.tight_layout()
    plt.savefig("images/cumdistributions.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

def plotMAPProbabilities(decay_factors, noise_levels, sampling_alg, simulation_dfs):

    fig, axs = plt.subplots(len(decay_factors), len(noise_levels), figsize=(8,6))

    if len(decay_factors) == 1 and len(noise_levels) == 1:

        # determine rule change points
        rule_change_indices = utils.getRuleChangeIndices(simulation_dfs[0][0])

        if len(rule_change_indices) >= 4:
            first_right_end = rule_change_indices[0]
            first_left_start = rule_change_indices[0] + 1
            first_left_end = rule_change_indices[1]
            second_right_start = rule_change_indices[1] + 1
            second_right_end = rule_change_indices[2]
            second_left_start = rule_change_indices[2] + 1
            second_left_end = rule_change_indices[3]
        else:
            # else handle exception
            print("Not enough rule changes detected.")

        # plot left and right probability
        axs.plot(simulation_dfs[0][0]['TrialIndex'], simulation_dfs[0][0]['MAP_GoRight'], label='Go Right', color='green')
        axs.plot(simulation_dfs[0][0]['TrialIndex'], simulation_dfs[0][0]['MAP_GoLeft'], label='Go Left', color='blue')

        # add vertical lines to indicate rule changes
        axs.axvline(x=first_right_end, color='grey', linestyle='--', lw=1)
        axs.axvline(x=first_left_end, color='grey', linestyle='--', lw=1)
        axs.axvline(x=second_right_end, color='grey', linestyle='--', lw=1)

        axs.axvspan(0, first_right_end, color='lightblue', alpha=0.3, label='go right 1')
        axs.axvspan(first_right_end, first_left_end, color='lightgreen', alpha=0.3, label='go left 1')
        axs.axvspan(first_left_end, second_right_end, color='lightcoral', alpha=0.3, label='go right 2')
        axs.axvspan(second_right_end, second_left_end, color='lightyellow', alpha=0.3, label='go left 2')

        # add labels and title
        axs.set_title(f"MAP Probability for Over-trained Agents During Simulation ($\\gamma={decay_factors[0]}$, $\\eta={noise_levels[0]}$)")
        axs.set_xlabel("Trial Index")
        axs.set_ylabel("MAP Probability")
        axs.set_ylim(0, 1)
        axs.axhline(y=0.5, color='grey', linestyle='--', linewidth=1)  # Horizontal line at y=0.5
        axs.legend(loc='lower right')
        axs.grid(True)
        axs.legend()

    else:
        for i, decay in enumerate(decay_factors):
            for j, noise_level in enumerate(noise_levels):

                # determine rule change points
                rule_change_indices = utils.getRuleChangeIndices(simulation_dfs[i][j])

                if len(rule_change_indices) >= 4:
                    first_right_end = rule_change_indices[0]
                    first_left_start = rule_change_indices[0] + 1
                    first_left_end = rule_change_indices[1]
                    second_right_start = rule_change_indices[1] + 1
                    second_right_end = rule_change_indices[2]
                    second_left_start = rule_change_indices[2] + 1
                    second_left_end = rule_change_indices[3]
                else:
                    # else handle exception
                    print("Not enough rule changes detected.")

                # plot MAP probabilities
                # plot go right probability
                axs[i][j].plot(simulation_dfs[i][j]['TrialIndex'], simulation_dfs[i][j]['MAP_GoRight'], label='Go Right', color='green')

                # plot go left probability
                axs[i][j].plot(simulation_dfs[i][j]['TrialIndex'], simulation_dfs[i][j]['MAP_GoLeft'], label='Go Left', color='blue')

                # add vertical lines to indicate rule changes
                axs[i][j].axvline(x=first_right_end, color='grey', linestyle='--', lw=1)
                axs[i][j].axvline(x=first_left_end, color='grey', linestyle='--', lw=1)
                axs[i][j].axvline(x=second_right_end, color='grey', linestyle='--', lw=1)

                axs[i][j].axvspan(0, first_right_end, color='lightblue', alpha=0.3, label='go right 1')
                axs[i][j].axvspan(first_right_end, first_left_end, color='lightgreen', alpha=0.3, label='go left 1')
                axs[i][j].axvspan(first_left_end, second_right_end, color='lightcoral', alpha=0.3, label='go right 2')
                axs[i][j].axvspan(second_right_end, second_left_end, color='lightyellow', alpha=0.3, label='go left 2')

                # add plot details
                axs[i][j].set_xlabel('Trial Index')
                axs[i][j].set_ylabel('Probability')
                axs[i][j].set_title(f'MAP Probabilities Over Time ($\\gamma={decay}$ , $\\eta={noise_level}$)', fontsize=12)
                axs[i][j].set_ylim(0, 1)
                axs[i][j].axhline(y=0.5, color='grey', linestyle='--', linewidth=1)  # Horizontal line at y=0.5
                axs[i][j].legend(loc='best')
                axs[i][j].grid(True)
        
    plt.tight_layout()
    plt.savefig("images/MAPprob.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

def plotTotalCorrect(decay_factors, noise_levels, sampling_alg, simulation_dfs):

    fig, axs = plt.subplots(len(decay_factors), len(noise_levels), figsize=(12,10))

    for i, decay in enumerate(decay_factors):
        for j, noise_level in enumerate(noise_levels):

            right_precision, left_precision = utils.calculate_totalcorrect(simulation_dfs[i][j])

            # plot precision for right and left rules
            axs[i, j].bar(['Right', 'Left'], [right_precision, left_precision], color=['blue', 'green'])
            axs[i, j].set_ylim(0, 1)  
            axs[i, j].set_title(f'Decay: {decay}, Noise: {noise_level}')
            axs[i, j].set_ylabel('% Correct of Right and Left Choices')
            axs[i, j].set_xlabel('Direction')

    # adjust layout and save figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
    plt.savefig("images/totalCorrect.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

def conductANOVA(unfiltered_data):

    unfiltered_data['Trials to Learn'] = pd.to_numeric(unfiltered_data['Trials to Learn'], errors='coerce')

    # run three-way ANOVA to see if there are differences in average learning due to decay factor, noise level or training condition
    model = ols('Q("Trials to Learn") ~ C(Q("Decay Factor")) * C(Q("Noise Level")) * C(Q("Overtrained"))', data=unfiltered_data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    # print anova test
    print("Two-way ANOVA on Decay Factor and Noise Level:")
    print(anova_table)

def plotStatistics(unfiltered_data):


    # separate overtrained from normally trained agents 
    overtrained_data = unfiltered_data[unfiltered_data['Overtrained'] == True]
    normal_data = unfiltered_data[unfiltered_data['Overtrained'] == False]

    # combine initial rule data for comparison
    initialrule_data = pd.concat([overtrained_data[overtrained_data['Rule'] == 'go right 1'], 
                                  normal_data[normal_data['Rule'] == 'go right 1']])
    initialrule_data['Training Type'] = initialrule_data['Overtrained'].map({True: 'Overtrained', False: 'Normal'})

    # combine first reversal data for comparison
    firstreversal_data = pd.concat([overtrained_data[overtrained_data['Rule'] == 'go left 1'], 
                                    normal_data[normal_data['Rule'] == 'go left 1']])
    firstreversal_data['Training Type'] = firstreversal_data['Overtrained'].map({True: 'Overtrained', False: 'Normal'})

    # combine second reversal data for comparison
    secondreversal_data = pd.concat([overtrained_data[overtrained_data['Rule'] == 'go right 2'], 
                                    normal_data[normal_data['Rule'] == 'go right 2']])
    secondreversal_data['Training Type'] = secondreversal_data['Overtrained'].map({True: 'Overtrained', False: 'Normal'})

    # combine third reversal data for comparison
    thirdreversal_data = pd.concat([overtrained_data[overtrained_data['Rule'] == 'go left 2'], 
                                    normal_data[normal_data['Rule'] == 'go left 2']])
    thirdreversal_data['Training Type'] = thirdreversal_data['Overtrained'].map({True: 'Overtrained', False: 'Normal'})

    # plot subplot of average trials to learn for normally vs overtrained agents for initial rule and first reversal
    fig, axs = plt.subplots(2, 2, figsize=(14,10))  

    # plot bar plots for each rule
    sns.barplot(x='Noise Level', y='Trials to Learn', hue='Training Type', data=initialrule_data, errorbar='se', palette='muted', ax=axs[0,0])
    axs[0,0].set_title('Mean Trials to Learn by Noise Level For Initial Rule ($\\gamma={0.9}$)')

    sns.barplot(x='Noise Level', y='Trials to Learn', hue='Training Type', data=firstreversal_data, errorbar='se', palette='muted', ax=axs[0,1])
    axs[0,1].set_title('Mean Trials to Learn by Noise Level For First Reversal ($\\gamma={0.9}$)')

    sns.barplot(x='Noise Level', y='Trials to Learn', hue='Training Type', data=secondreversal_data, errorbar='se', palette='muted', ax=axs[1,0])
    axs[1,0].set_title('Mean Trials to Learn by Noise Level For Second Reversal ($\\gamma={0.9}$)')

    sns.barplot(x='Noise Level', y='Trials to Learn', hue='Training Type', data=thirdreversal_data, errorbar='se', palette='muted', ax=axs[1,1])
    axs[1,1].set_title('Mean Trials to Learn by Noise Level For Third Reversal ($\\gamma={0.9}$)')

    # set axis labels
    for ax in axs.flat:
        ax.set(xlabel='Noise Level', ylabel='Mean Trials to Learn')

    plt.tight_layout()  
    plt.savefig("images/statisticsplot.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
