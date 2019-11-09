from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np 
import matplotlib.pyplot as plt
from gpts import CombinatorialGPTSLearner, GPTSLearner
from tqdm import tqdm




def print_this_array_with_precision_2(what_to_print, bidding_environment=None, precision=2):
    if bidding_environment is not None:
        return [(idx, bidding_environment.get_subcampaign(idx).get_functions_ids(),
             float("{0:.2f}".format(arm_bid))) for (idx, arm_bid) in what_to_print]
    return [(idx, float("{0:.2f}".format(arm_bid))) for (idx, arm_bid) in what_to_print] 

def print_clairvoyants(bidding_environment, aggregated_solution=None, aggregated_reward=None, disaggregated_solution=None, disaggregated_reward=None, print_combination=False):
    if print_combination:
        print(print_this_array_with_precision_2(aggregated_solution, bidding_environment=bidding_environment))
    print('Aggregated optimal reward  ==>  ', aggregated_reward)
    if print_combination:
        print(print_this_array_with_precision_2(disaggregated_solution))
    print('Disaggregated optimal reward  ==>  ', disaggregated_reward)

def create_partitions(collection):
    if len(collection) == 1:
        yield [ collection ]
        return
    first = collection[0]
    for smaller in create_partitions(collection[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        yield [ [ first ] ] + smaller




# This is the function that disaggregates the subcampaigns based on the observed values of the past. 
# Basically, what we do is creating a number of BiddingEnvironments equal to the number of the possible
# ways of disaggregating the subcampaigns. 
# For each BiddingEnviroment then, we create a CombinatorialLearner containing a number of GPs equal
# to the number of subcampaigns of the specific environment taken into account.
# After retrieving the already observed samples from the real enviroment and training all the GPs in the
# Combinatorial, we .predict() and get some samples that will be again given to the MultiChoiceKnapsack
# to understand which combination is the best one. Finally, if the reward returned observing the real curves
# is better than the current one, we mark this enviroment as the best. Meaning that the way it splitted
# the subcampaigns is the current best one. 
# In the end, we return the best environment and combinatorial_learner that have been created during this process,
# so that they can be used in the future iterations of the program (i.e., if we have disaggregated in some way
# we will keep it disaggregated in this way until the next context-identification).
def disaggregate_subcampaigns(bidding_environment, daily_budget, arms_bids, num_disaggregated_subcampaigns=5):

    all_disaggregations = [[]]
    environments_to_analyze = np.power(5, num_disaggregated_subcampaigns)

    for reference_subcampaign in bidding_environment.sub_campaigns:
        possible_partitions = reference_subcampaign.partition()
        temp_disaggregation = []
        for disaggregation in all_disaggregations:
            for part in possible_partitions:
                temp_disaggregation.append([d.copy() for d in disaggregation] + [p.copy() for p in part])
        all_disaggregations = temp_disaggregation.copy()

    all_disaggregated_environments = []

    for disaggregation in all_disaggregations:

        sub_campaigns = []
        new_subcampaign_id = 1

        for d_subcampaign in disaggregation:

            functions = []
            already_collected_samples = []
            already_pulled_arms = []

            for func_idx in d_subcampaign:

                reference_subcampaign = bidding_environment.get_subcampaign_with_function_index(func_idx)
                functions.append(reference_subcampaign.get_function(func_idx))
                already_collected_samples.append(reference_subcampaign.get_function_collected_samples(func_idx).copy())
                already_pulled_arms = reference_subcampaign.pulled_arms.copy()
            
            sub_campaigns.append(Subcampaign(functions, sub_campaign_id=new_subcampaign_id, func_numbers=tuple(d_subcampaign),
                        already_observed_samples=already_collected_samples, already_pulled_arms=already_pulled_arms))
            new_subcampaign_id += 1
        
        all_disaggregated_environments.append(BiddingEnvironment(sub_campaigns))


    best_disaggregated_environment = None
    best_disaggregated_combinatorial_learner = None
    best_total_reward = -np.inf

    for disaggregated_environment in tqdm(all_disaggregated_environments[:environments_to_analyze]):
        num_subcampaigns = disaggregated_environment.num_subcampaigns

        gpts_subproblems = [GPTSLearner(len(arms_bids), arms_bids, idx) for idx in range(1, num_subcampaigns+1)]
        disaggregated_combinatorial_learner = CombinatorialGPTSLearner(gpts_subproblems)

        for sub_campaign_id in range(1, disaggregated_environment.num_subcampaigns+1):
            sub_camp = disaggregated_environment.get_subcampaign(sub_campaign_id)
            for i in range(len(sub_camp.pulled_arms)):
                for j in sub_camp.get_functions_ids():
                    disaggregated_combinatorial_learner.get_gpts_subproblem(sub_campaign_id).update_observations(sub_camp.pulled_arms[i], 
                                    sub_camp.get_function_collected_samples(j)[i])
            disaggregated_combinatorial_learner.get_gpts_subproblem(sub_campaign_id).update_model()
        
        disaggregation_samples = disaggregated_combinatorial_learner.predict()
        disaggregation_super_arm = MultiChoiceKnapsack(disaggregation_samples, daily_budget, arms_bids).find_optimal_config()

        disaggregation_total_reward = disaggregated_combinatorial_learner.get_total_reward(disaggregation_super_arm, disaggregated_environment)

        if disaggregation_total_reward > best_total_reward:
            best_total_reward = disaggregation_total_reward
            best_disaggregated_environment = disaggregated_environment.copy()
            best_disaggregated_combinatorial_learner = disaggregated_combinatorial_learner
        
    print('New disaggregation ==> # Subcampaigns ', best_disaggregated_environment.num_subcampaigns, '  -  Total reward ', best_total_reward, '\n\n\n')
    
    return best_disaggregated_combinatorial_learner, best_disaggregated_environment
















# The Bidding Environment contains all subcampaigns and with them the functions that compose them. It is
# basically the container of all subcampaigns.
class BiddingEnvironment():
    def __init__(self, sub_campaigns):
        self.sub_campaigns = sub_campaigns
        self.num_subcampaigns = len(self.sub_campaigns)
    def get_subcampaign(self, index):
        for sc in self.sub_campaigns:
            if sc.get_subcampaign_id() == index:
                return sc
        return None
    def get_subcampaign_with_function_index(self, func_index):
        for sub_camp in self.sub_campaigns:
            for fidx in sub_camp.get_functions_ids():
                if fidx == func_index:
                    return sub_camp
        return None
    def copy(self):
        return BiddingEnvironment([sub_camp.copy() for sub_camp in self.sub_campaigns])








# Each Subcampaign object has functions associated with it and, when we sample its value after giving a budget to it,
# we divide the chosen budget within the 1, 2 or 3 different functions and return the result.
# The way we designed the class, Subcampaign can contain also just one function representing a specific class of users.
# The functions inside the subcampaign are associated with an id, as it can be seen in self.func_ids.
# The samples collected are already divided between the different functions to speed up the computation when we 
# need to disaggregate.
class Subcampaign():
    def __init__(self, functions, sub_campaign_id=-1, func_numbers=None, already_observed_samples=None, already_pulled_arms=None):
        self.functions = functions
        self.num_functions = len(self.functions)
        self.std = 0.2
        self.sc_id = sub_campaign_id
        if func_numbers is None:
            self.func_ids = tuple(range(1, self.num_functions+1))
        else: 
            self.func_ids = func_numbers

        if already_observed_samples is None:
            self.collected_samples = [[] for _ in range(self.num_functions)]
            self.pulled_arms = []
        else:
            self.collected_samples = already_observed_samples
            self.pulled_arms = already_pulled_arms

    def sample(self, arm_bid, avg=False):
        noise = np.random.normal(0, self.std)
        if avg:
            samples = [(func (arm_bid) / self.num_functions) + noise for func in self.functions]
        else:
            samples = [func( arm_bid / self.num_functions ) + noise for func in self.functions]
        for i in range(self.num_functions):
            self.collected_samples[i].append(samples[i] * len(samples))
        self.pulled_arms.append(arm_bid)
        return samples

    def get_disaggregated_curves(self, arm_bid):
        return [func (arm_bid) for func in self.functions]

    def get_real_values(self, arms_bids):
        result = np.zeros(len(arms_bids))
        for func in self.functions:
            result = result + func( arms_bids / self.num_functions )
        return result
    
    def get_real_avg(self, arms_bids):
        result = np.zeros(len(arms_bids))
        for func in self.functions:
            result = result + func( arms_bids ) / self.num_functions
        return result

    def get_arm_real_value(self, arm_bid):
        result = 0.0
        for func in self.functions:
            result = result + func( arm_bid / self.num_functions ) 
        return result
    
    def get_arm_real_avg(self, arm_bid):
        result = 0.0
        for func in self.functions:
            result = result + func( arm_bid ) / self.num_functions
        return result

    def partition(self):
        funcs_indexes = list(self.get_functions_ids())
        subcampaign_partitions = create_partitions(funcs_indexes)
        possible_partitions = []
        for sp in subcampaign_partitions:
            possible_partitions.append(sp)
        return possible_partitions

    def get_subcampaign_id(self):
        return self.sc_id

    def get_functions_ids(self):
        return self.func_ids

    def get_function(self, index):
        for idx in range(self.num_functions):
            if self.func_ids[idx] == index:
                return self.functions[idx]
        return None
    
    def get_function_collected_samples(self, index):
        for idx in range(self.num_functions):
            if self.func_ids[idx] == index:
                return self.collected_samples[idx]
        return None

    def plot(self, dis=False):
        plt.figure()
        x = np.arange(0, 1.05, 0.05)
        y = [self.get_disaggregated_curves(i) for i in x]
        plt.plot(x, y)
        y_real = self.get_real_values(x)
        y_avg = self.get_real_avg(x)
        plt.plot(x, y_real, 'r')
        plt.plot(x, y_avg, 'k--')
        if dis:
            plot_path = 'plots/curves/' + str(self.get_subcampaign_id()) + '-SUBCAMPAIGN_DISAG.png'
        else:
            plot_path = 'plots/curves/' + str(self.get_subcampaign_id()) + '-SUBCAMPAIGN.png'
        plt.savefig(plot_path, bbox_inches='tight', dpi = 300)
        plt.close()

    def copy(self, new_id=-1):
        if new_id < 0:
            new_id = self.sc_id
        return Subcampaign(self.functions, sub_campaign_id=new_id, func_numbers=self.func_ids,
            already_observed_samples=self.collected_samples, already_pulled_arms=self.pulled_arms)













# Experiment is the most important class of the project. Through the .fit() method we can choose if during the days of computation
# we want to split context-wise or not, we can choose the interval to do so, and set up all arrays that we need to collect the data
# that we will need to print later on.

# At the beginning, we create one GPTS object for each subcampaign in the BiddingEnviroment object passed to the class and then
# we pass all of them to the CombinatorialGPTSLearner that will do almost all the job: .predict() samples based on every single GPTS,
# .observe_and_update() each GPTS after observing the real samples through the support of the actual bidding_enviroment and finally 
# get the regression error of the chosen subcampaign, if needed.
# Thanks to the MultiChoiceKnapsack class, we can find the best super_arm to pull. It will look like these:
# [     (subcampaign_id,    (func_1_id, func_2_id, ...),    budget_for_subcampaign),      (...), ...]. 
# After applying the optimization algorithm, we check if it's the first day of computation: in this case we know nothing
# about the users (i.e. GPs are not trained yet...), so we split the budget between the 5 subcampaigns equally.
class Experiment():

    def __init__(self, arms_bids, daily_budget, num_iterations=70):
        self.arms_bids = arms_bids
        self.daily_budget = daily_budget
        self.num_iterations = num_iterations
    
    def fit(self, bidding_environment, clairvoyant_reward, subcampaign_id_regression_error=-1, contextualise=False, context_analysis_interval=-1):
        self.bidding_environment = bidding_environment
        self.contextualise = contextualise
        self.context_analysis_interval = context_analysis_interval
        self.rewards = []
        self.clairvoyant_reward = clairvoyant_reward
        self.regret = []
        self.subcampaign_id_regression_error = subcampaign_id_regression_error
        self.chosen_subcampaign_regression_error = []

        gpts_subproblems = [GPTSLearner(len(self.arms_bids), self.arms_bids, idx) for idx in range(1, self.bidding_environment.num_subcampaigns+1)]
        self.combinatorial_learner = CombinatorialGPTSLearner(gpts_subproblems)

    def start(self):

        bidding_environment_copy = self.bidding_environment.copy()
        
        for t in range(self.num_iterations):

            samples = self.combinatorial_learner.predict()
            super_arm = MultiChoiceKnapsack(samples, self.daily_budget, self.arms_bids).find_optimal_config()

            if t == 0: 
                super_arm = [(sub_camp, self.daily_budget / bidding_environment_copy.num_subcampaigns) for sub_camp in range(1, bidding_environment_copy.num_subcampaigns+1)]
            
            print('DAY ' + str(t) + ' ==> ', print_this_array_with_precision_2(super_arm, bidding_environment=bidding_environment_copy))

            # .observe_and_update() method takes the super_arm chosen at this round and observes the results coming from
            # the subcampaigns real functions applying the budget division as stated in the super_arm.
            # example of super_arm: [(1, (1, 2, 3), 0.2), (2, (4, 5, 6), 0.2), (3, (7, 8, 9), 0.2), (4, (10, 11, 12), 0.2), (5, (13, 14, 15), 0.2)]
            # We observe the reward we get putting 0.2 on the 5th subcampaign, 0.2 on the 4th, 0.2 on the 3rd...
            # Moreover, it updates the GPTS learners of the combinatorial using the observed results.
            total_reward = self.combinatorial_learner.observe_and_update(super_arm, bidding_environment_copy)
            # Here we collect rewards and regret of each iteration so that we can print them out later on to show the
            # difference between the 2 cases.
            self.rewards.append(total_reward)
            self.regret.append(self.clairvoyant_reward - total_reward)

            # We enter in this 'if' only during the computation the aggregated case and it's needed to collect the regression error
            # values that will later on be printed.
            if self.subcampaign_id_regression_error > 0:
                self.chosen_subcampaign_regression_error.append(
                    self.combinatorial_learner.get_subcampaign_regression_error(self.subcampaign_id_regression_error, 
                        bidding_environment_copy.get_subcampaign(self.subcampaign_id_regression_error)))

            print(total_reward)

            # We enter in this 'if' only during the context identification process...
            if self.contextualise and (t+1) % self.context_analysis_interval == 0:

                print('\n\nWeekly context algorithm applied. Looking for better disaggregation...\n\n')

                # putting num_disaggregated_subcampaigns = 
                # 1 -> analizziamo 5 possibili disaggregazioni, cioè valutiamo solo se conviene disaggregare la subcampaign 5
                # 2 -> analizziamo 25 possibili disaggregazioni, cioè valutiamo se conviene disaggregare la subcampaign 4 e 5
                # 3 -> analizziamo 125 possibili disaggregazioni, cioè valutiamo se conviene disaggregare la subcampaign 3, 4 e 5
                # ... 5 -> we analyze 3125 ... ATTENTION: long wait, 3 suggested for simple run!
                self.combinatorial_learner, bidding_environment_copy = disaggregate_subcampaigns(bidding_environment_copy, 
                                                                        self.daily_budget, self.arms_bids, num_disaggregated_subcampaigns=5)

        for sp in self.combinatorial_learner.gpts_subproblems:
            sp.plot(dis=self.contextualise)
        for sc in bidding_environment_copy.sub_campaigns:
            sc.plot(dis=self.contextualise)








class MultiChoiceKnapsack():

    def __init__(self, observed_values, daily_budget, arms_bids):
        self.daily_budget = daily_budget
        self.observed_values = observed_values
        self.arms = arms_bids

    # Here we just put down to code what has been shown in the lecture's slides.
    def find_optimal_config(self):
        merged_results = [[0.0]*len(self.arms)]
        merged_indices = []

        for row in self.observed_values:
            tmp_results = []
            tmp_indices = []
            last_merged_row = merged_results[-1]
            for i in range(len(row)):
                optimal_value = 0.0
                optimal_indices = (i,0)
                for j in range(0, i+1):
                    index = i - j
                    if(row[index] + last_merged_row[j] > optimal_value):
                        optimal_value = row[index] + last_merged_row[j]
                        optimal_indices = (index, j)
                tmp_results.append(optimal_value)
                tmp_indices.append(optimal_indices)
            merged_results.append(tmp_results)
            merged_indices.append(tmp_indices)
        
        bottom_right = merged_indices[-1][-1]

        return self.find_combination(bottom_right, merged_indices.copy())

    def find_combination(self, best_row_comb, combinations, result=None):
        if result is None:
            result = []
        result.append((len(combinations), self.arms[best_row_comb[0]]))
        combinations.pop()
        if len(combinations) == 0:
            return result
        best_row_comb = combinations[-1][best_row_comb[1]]
        return self.find_combination(best_row_comb, combinations, result)


































