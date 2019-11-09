import numpy as np 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ccd
import matplotlib.pyplot as plt
from utilities import BiddingEnvironment, Subcampaign, MultiChoiceKnapsack, Experiment, print_clairvoyants
from functions import get_functions









print_functions = True

##########################
###  BUILDING PROGRAM  ###
##########################

T = 70
num_arms = 20
daily_budget = 1
discrete_bid_per_arm = np.linspace(0, daily_budget, num_arms + 1, dtype=float)

classes_functions = get_functions()

sub_campaigns = []
i = 1
sc_id = 1
a = [1, 1, 1]
for sub_functions in classes_functions:
    sub_campaigns.append( Subcampaign( sub_functions, sub_campaign_id=sc_id, func_numbers=tuple([a[j]*i + j for j in range(len(a))]) ) )
    i += 3
    sc_id += 1
bidding_environment = BiddingEnvironment(sub_campaigns)






##################################
###  PRINT ORIGINAL FUNCTIONS  ###
##################################
# In this part of the program we print the aggregated functions, representing the subcampaigns
# and the disaggregated functions, each one of them representing a class of users.

if print_functions:
    
    x = np.arange(0, 1.05, 0.05)
    
    plt.figure()
    legend = []
    for sub_campaign in bidding_environment.sub_campaigns:
        legend.append('Subcampaign bid split ' + str(sub_campaign.get_subcampaign_id()))
        plt.plot(x, sub_campaign.get_real_values(x), '--')
        legend.append('Subcampaign avg ' + str(sub_campaign.get_subcampaign_id()))
        plt.plot(x, sub_campaign.get_real_avg(x))
    plt.legend(legend, bbox_to_anchor = (1.05, 1), loc = 2)
    plt.title('Aggregated subcampaigns')
    plt.savefig('plots/aggregated.png', bbox_inches='tight', dpi = 300)
    plt.close()

    plt.figure()
    legend = []
    func_index = 0
    for sub_campaign in bidding_environment.sub_campaigns:
        for func in sub_campaign.functions:
            legend.append('Function ' + str(sub_campaign.get_functions_ids()[func_index]))
            func_index += 1
            plt.plot(x, func(x))
        func_index = 0
    plt.legend(legend, bbox_to_anchor = (1.05, 1), loc = 2)
    plt.title('Disaggregated subcampaigns')
    plt.savefig('plots/disaggregated.png', bbox_inches='tight', dpi = 300)
    plt.close()










#############################
### CLAIRVOYANT SOLUTION  ###
#############################
# In this part, we sample the real values of every subcampaign that we have created
# and we pass the array to the MultiChoiceKnapsack where, through the find_optimal_config()
# method, we can find the best way to distribute our budget within the 5 subcampaigns, getting to
# know the optimal reward.
# In the second part of this section, we do the same but with the disaggregated curves
# so that we can know the optimal reward in the case in which we know all classes of users.

real_observations = []
for sub_campaign in bidding_environment.sub_campaigns:
    real_observations.append(sub_campaign.get_real_values(discrete_bid_per_arm))
optimal_super_arm = MultiChoiceKnapsack(real_observations, daily_budget, discrete_bid_per_arm).find_optimal_config()
optimal_super_arm_value = sum([bidding_environment.get_subcampaign(i).get_arm_real_value(arm) for (i, arm) in optimal_super_arm])

real_observations = []
funcs_dict = dict()
funcs_index = 0
for sub_campaign in bidding_environment.sub_campaigns:
    for func in sub_campaign.functions:
        real_observations.append(func(discrete_bid_per_arm))
        funcs_dict[sub_campaign.get_functions_ids()[funcs_index]] = func
        funcs_index += 1
    funcs_index = 0
optimal_disaggregated_super_arm = MultiChoiceKnapsack(real_observations, daily_budget, discrete_bid_per_arm).find_optimal_config()
optimal_disaggregated_super_arm_value = sum([funcs_dict[i](arm) for (i, arm) in optimal_disaggregated_super_arm])

print_clairvoyants(bidding_environment, print_combination=False, aggregated_solution=optimal_super_arm, disaggregated_solution=optimal_disaggregated_super_arm, 
    aggregated_reward=optimal_super_arm_value, disaggregated_reward=optimal_disaggregated_super_arm_value)







#####################
###  EXPERIMENTS  ### 
#####################
# In this part we create two different objects: one for the experiment without context identification and 
# one for the experiment with context identification. Naturally, context identification is done supposing
# to have found 3 classes of users for each subcampaign, and these classes are represented by the functions 
# of the specific subcampaign.
# - num_iterations is the number of days that the experiment lasts
# - subcampaign_id_regression_error=3 tells for which subcampaign we need to collect the regression error (as
#   stated in point 3.4 of the project) that we will print later on.
# - contextualise tells the experiment if we want to use the 'context identification algorithm'
# - context_analysis_interval tells the interval between the application of the context identification algorithm
# Inside the experiment we create the CombinatorialGPTSLearner that will be used to observe the different GPTS learners
# and choose in which arms is better to put our daily_budget.

print('\n\n #####  EXPERIMENT WITHOUT CONTEXT IDENTIFICATION  #####\n')

aggregated_experiment = Experiment(discrete_bid_per_arm, daily_budget, num_iterations=T)
aggregated_experiment.fit(bidding_environment.copy(), optimal_super_arm_value, subcampaign_id_regression_error=3,
    contextualise=False, context_analysis_interval=-1)
aggregated_experiment.start()

print('\n\n #####  EXPERIMENT WITH CONTEXT IDENTIFICATION  #####\n')

# If faster computation is needed, you can restrict the context identification to just 4, 3, 2 or 1 subcampaign.
# The standard is analyzing all 5 subcampaigns. The total reward will suffer a decrease in value, though.
# Check out the variable num_disaggregated_subcampaigns in the function utilities.disaggregate_subcampaigns().

disaggregated_experiment = Experiment(discrete_bid_per_arm, daily_budget, num_iterations=T)
disaggregated_experiment.fit(bidding_environment.copy(), optimal_disaggregated_super_arm_value, 
    contextualise=True, context_analysis_interval=7)
disaggregated_experiment.start()










###########################################
###  PRINT REGRET FOR BOTH EXPERIMENTS  ###
###########################################
# Printing of Regret as requested in points 3.3 and 3.5, comparing the one belonging to the aggregated case
# with the one belonging to the disaggregated case.

legend = []
plt.figure()
plt.xlabel('Time')
plt.ylabel('Regret')
legend.append('Regret of Aggregated solution')
plt.plot(aggregated_experiment.regret, 'r')
legend.append('Regret of Disaggregated solution')
plt.plot(disaggregated_experiment.regret, 'b')
plt.legend(legend, bbox_to_anchor = (1.05, 1), loc = 2)
plt.title('Regret\'s variation in time')
plt.savefig('plots/REGRET.png', bbox_inches='tight', dpi = 300)
plt.close()

###########################################
###  PRINT REWARD FOR BOTH EXPERIMENTS  ###
###########################################
# Printing of Reward as requested in points 3.5, comparing the one belonging to the aggregated case
# with the one belonging to the disaggregated case.

legend = []
plt.figure()
plt.xlabel('Time')
plt.ylabel('Reward')
legend.append('Rewards of Aggregated solution')
plt.plot(aggregated_experiment.rewards, 'r')
legend.append('Reward of Disaggregated solution')
plt.plot(disaggregated_experiment.rewards, 'b')
plt.legend(legend, bbox_to_anchor = (1.05, 1), loc = 2)
plt.title('Reward\'s variation in time')
plt.savefig('plots/REWARD.png', bbox_inches='tight', dpi = 300)
plt.close()

##################################################
###  PRINT REGRESSION ERROR FOR SUBCAMPAIGN 3  ###
##################################################
# Printing of Regression Error for subcampaign 3 (the one we have chosen to analyze).

plt.figure()
plt.xlabel('Number of Samples')
plt.ylabel('Regression Error')
plt.plot(aggregated_experiment.chosen_subcampaign_regression_error, 'r')
plt.title('Regression Error for subcampaign 3')
plt.savefig('plots/REGRESSION-ERROR-Subcampaign3.png', bbox_inches='tight', dpi = 300)
plt.close()


