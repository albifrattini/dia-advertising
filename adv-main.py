import numpy as np 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ccd
import matplotlib.pyplot as plt
from utilities import BiddingEnvironment, Subcampaign, MultiChoiceKnapsack, Experiment, print_this_array_with_precision_2

from functions import get_functions



print_functions = True

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






#############################
## PRINT ORIGINAL FUNCTIONS
#############################

if print_functions:
    
    x = np.arange(0, 1.05, 0.05)
    
    plt.figure()
    legend = []
    # Aggregated subcampaigns
    for sub_campaign in bidding_environment.sub_campaigns:
        legend.append('Subcampaign bid split ' + str(sub_campaign.get_subcampaign_id()))
        plt.plot(x, sub_campaign.get_real_values(x), '--')
        legend.append('Subcampaign avg ' + str(sub_campaign.get_subcampaign_id()))
        plt.plot(x, sub_campaign.get_real_avg(x))
    plt.legend(legend, bbox_to_anchor = (1.05, 1), loc = 2)
    plt.title('Aggregated subcampaigns')
    plt.savefig('/Users/alberto/development/dia/advertising/plots/aggregated.png', bbox_inches='tight', dpi = 300)
    plt.close()

    plt.figure()
    legend = []
    func_index = 0
    # Fully disaggregated subcampaigns
    for sub_campaign in bidding_environment.sub_campaigns:
        for func in sub_campaign.functions:
            legend.append('Function ' + str(sub_campaign.get_functions_ids()[func_index]))
            func_index += 1
            plt.plot(x, func(x))
        func_index = 0
    plt.legend(legend, bbox_to_anchor = (1.05, 1), loc = 2)
    plt.title('Disaggregated subcampaigns')
    plt.savefig('/Users/alberto/development/dia/advertising/plots/disaggregated.png', bbox_inches='tight', dpi = 300)
    plt.close()







############################
## CLAIRVOYANT
############################

real_observations = []
for sub_campaign in bidding_environment.sub_campaigns:
    real_observations.append(sub_campaign.get_real_values(discrete_bid_per_arm))
optimal_super_arm = MultiChoiceKnapsack(real_observations, daily_budget, discrete_bid_per_arm).find_optimal_config()
print(print_this_array_with_precision_2(optimal_super_arm, bidding_environment=bidding_environment))
optimal_super_arm_value = sum([bidding_environment.get_subcampaign(i).get_arm_real_value(arm) for (i, arm) in optimal_super_arm])
print(optimal_super_arm_value)

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
print(print_this_array_with_precision_2(optimal_disaggregated_super_arm))
optimal_disaggregated_super_arm_value = sum([funcs_dict[i](arm) for (i, arm) in optimal_disaggregated_super_arm])
print(optimal_disaggregated_super_arm_value)

clairvoyant_rewards = [optimal_super_arm_value for _ in range(T)]
clairvoyant_disaggregated_rewards = [optimal_disaggregated_super_arm_value for _ in range(T)]



exit()

###########################
## EXPERIMENT
###########################

experiment = Experiment(discrete_bid_per_arm, daily_budget, num_iterations=T)


print('-------- GPTS ---------')

experiment.fit(bidding_environment.copy(), contextualise=False, context_analysis_interval=-1)
experiment.start()


print('-------- GPTS context --------')

experiment.fit(bidding_environment.copy(), contextualise=True, context_analysis_interval=7)
experiment.start()