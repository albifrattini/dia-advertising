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

def create_partitions(collection):
    if len(collection) == 1:
        yield [ collection ]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert first in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put first in its own subset 
        yield [ [ first ] ] + smaller

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
            # print('Temp best: num of subcampaigns -> ', disaggregated_environment.num_subcampaigns, ' total reward -> ', disaggregation_total_reward)
            best_disaggregated_environment = disaggregated_environment.copy()
            best_disaggregated_combinatorial_learner = disaggregated_combinatorial_learner
        
    print('\nNew disaggregation ==> # Subcampaigns ', best_disaggregated_environment.num_subcampaigns, '  -  Total reward ', best_total_reward, '\n')
    
    return best_disaggregated_combinatorial_learner, best_disaggregated_environment

















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
            # divide the bid between the available functions
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
            # divide the bid between the available functions
            result = result + func( arm_bid / self.num_functions ) 
        return result
    
    def get_arm_real_avg(self, arm_bid):
        result = 0.0
        for func in self.functions:
            # divide the bid between the available functions
            result = result + func( arm_bid ) / self.num_functions
        return result

    def partition(self):
        funcs_indexes = list(self.get_functions_ids())
        subcampaign_partitions = create_partitions(funcs_indexes)
        possible_partitions = []
        for sp in subcampaign_partitions:
            possible_partitions.append(sp)
        return possible_partitions


    def copy(self, new_id=-1):
        if new_id < 0:
            new_id = self.sc_id
        return Subcampaign(self.functions, sub_campaign_id=new_id, func_numbers=self.func_ids,
            already_observed_samples=self.collected_samples, already_pulled_arms=self.pulled_arms)

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

    def to_string(self):
        return '( ' + str(self.sc_id) + ' = ' + str(self.func_ids) + ' )'

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
            plot_path = '/Users/alberto/development/dia/advertising/plots/subcampaign_disag_' + str(self.get_subcampaign_id()) + '_real.png'
        else:
            plot_path = '/Users/alberto/development/dia/advertising/plots/subcampaign_' + str(self.get_subcampaign_id()) + '_real.png'
        plt.savefig(plot_path, bbox_inches='tight', dpi = 300)
        plt.close()














class Experiment():

    def __init__(self, arms_bids, daily_budget, num_iterations=70):
        self.arms_bids = arms_bids
        self.daily_budget = daily_budget
        self.num_iterations = num_iterations
    
    def fit(self, bidding_environment, contextualise=False, context_analysis_interval=-1):
        self.bidding_environment = bidding_environment
        self.contextualise = contextualise
        self.context_analysis_interval = context_analysis_interval

        gpts_subproblems = [GPTSLearner(len(self.arms_bids), self.arms_bids, idx) for idx in range(1, self.bidding_environment.num_subcampaigns+1)]
        self.combinatorial_learner = CombinatorialGPTSLearner(gpts_subproblems)

        if not contextualise:
            self.regression_error_sum = [[] for _ in range(self.bidding_environment.num_subcampaigns)]
    
    def plot(self):
        for i in range(1, len(self.combinatorial_learner.gpts_subproblems)+1):
            self.combinatorial_learner.get_gpts_subproblem(i).plot()
            self.bidding_environment.get_subcampaign(i).plot()

    def start(self):

        rewards = []
        bidding_environment_copy = self.bidding_environment.copy()
        
        for t in range(self.num_iterations):

            # predict method allows to retrieve samples based on the GPTS learners that represent the 5 subcampaigns
            # actually we are simply getting the predicted_values for each gpts class
            samples = self.combinatorial_learner.predict()
            super_arm = MultiChoiceKnapsack(samples, self.daily_budget, self.arms_bids).find_optimal_config()

            # the first day we divide the whole budget between the 5 subcampaigns
            if t == 0: 
                super_arm = [(sub_camp, self.daily_budget / bidding_environment_copy.num_subcampaigns) for sub_camp in range(1, bidding_environment_copy.num_subcampaigns+1)]
                # TODO: round to nearest possible bid
            
            print('DAY --> ' + str(t) + ' ==> ', print_this_array_with_precision_2(super_arm, bidding_environment=bidding_environment_copy))

            # observe_and_update method takes the super_arm chosen at this round and observes the results applying the budget
            # division as stated in the super_arm
            # example: [(4, 0.05), (3, 0.0), (2, 0.45), (1, 0.25), (0, 0.25)]
            # we will now observe the reward we get putting 0.05 on the 5th subcampaign, 0.0 on the 4th, 0.45 on the 3rd...
            # moreover, it updates the GPTS learners using the observed results
            total_reward = self.combinatorial_learner.observe_and_update(super_arm, bidding_environment_copy)
            print(total_reward)
            rewards.append(total_reward)

            if self.contextualise and (t+1) % self.context_analysis_interval == 0:

                print('\n\nWeekly context algorithm applied. Looking for better disaggregation...\n\n')

                # mettendo num_disaggregated_subcampaigns = 
                # 1 -> analizziamo 5 possibili disaggregazioni, cioè valutiamo solo se conviene disaggregare la subcampaign 5
                # 2 -> analizziamo 25 possibili disaggregazioni, cioè valutiamo se conviene disaggregare la subcampaign 4 e 5
                # 3 -> analizziamo 125 possibili disaggregazioni, cioè valutiamo se conviene disaggregare la subcampaign 3, 4 e 5
                # ... 5 -> analizziamo 3125 ... ATTENZIONE: lunga durata, 3 consigliato!
                self.combinatorial_learner, bidding_environment_copy = disaggregate_subcampaigns(bidding_environment_copy, self.daily_budget, self.arms_bids, num_disaggregated_subcampaigns=3)

        for sp in self.combinatorial_learner.gpts_subproblems:
            sp.plot(dis=self.contextualise)
        for sc in bidding_environment_copy.sub_campaigns:
            sc.plot(dis=self.contextualise)








class MultiChoiceKnapsack():

    def __init__(self, observed_values, daily_budget, arms_bids):
        self.daily_budget = daily_budget
        self.observed_values = observed_values
        self.arms = arms_bids

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
        # TODO: riscrivere funzione perchè troppo uguale a quella dei tizi
        if result is None:
            result = []
        result.append((len(combinations), self.arms[best_row_comb[0]]))
        combinations.pop()
        if len(combinations) == 0:
            return result
        best_row_comb = combinations[-1][best_row_comb[1]]
        return self.find_combination(best_row_comb, combinations, result)




































def disaggregate_subcampaign(bidding_environment, daily_budget, arms_bids, sub_campaign_id=1):
    # subcampaign che andremo a disaggregare nei 5 modi possibili e di cui valuteremo l'utilizzo della disaggregazione
    reference_subcampaign = bidding_environment.get_subcampaign(sub_campaign_id)
    # = list(0, 1, 2)
    # funcs_indexes = list(range(1, reference_subcampaign.num_functions+1))
    funcs_indexes = list(reference_subcampaign.get_functions_ids())
    possible_partitions = partition(funcs_indexes)

    temp_environment = None
    all_disaggregations = []

    for pp in possible_partitions:
        
        # pp = [[0, 1, 2]] || [[0], [1, 2]] || [[0, 1], [2]] || ...
        # se la partizione corrisponde a [0, 1, 2], cioè la 3 curve insieme, manteniamo lo stesso environment
        # di partenza
        if len(pp[0]) == reference_subcampaign.num_functions:
            temp_environment = bidding_environment.copy()
        # se invece la partizione è [0], [1, 2], per esempio, andiamo a creare un nuovo environment dove le sub_campaigns
        # diverse da quella con indice = sub_campaign_id rimangono le stesse, mentre quella interessata viene suddivisa secondo
        # la specifica, creando varie Subcampaign e infine un BiddingEnvironment
        # TODO: devo poter importare i sample osservati fino ad ora nella subcampaign per trainare i vari GP
        else:
            new_sub_campaign_id = 1
            sub_campaigns = []
            # for pp = [[0], [1, 2]] -> i = [0] and then [1, 2]
            for i in pp:
                functions = []
                already_collected_samples = []
                already_pulled_arms = []
                # for pp = [[0], [1, 2]] AND i = [0] and then [1, 2] -> j = 0 and then 1, 2
                for j in i:
                    functions.append(reference_subcampaign.get_function(j))
                    already_collected_samples.append(reference_subcampaign.get_function_collected_samples(j).copy())
                    already_pulled_arms = reference_subcampaign.pulled_arms.copy()
                # I create a new subcampaign with the already observed samples and their arms in the past rounds
                # Only the collected samples belonging to the specific function of the subcampaign created are 
                # passed to the new object.
                sub_campaigns.append(Subcampaign(functions, sub_campaign_id=new_sub_campaign_id, func_numbers=tuple(i),
                        already_observed_samples=already_collected_samples, already_pulled_arms=already_pulled_arms))
                new_sub_campaign_id += 1
            for i in range(1, bidding_environment.num_subcampaigns+1):
                if i != sub_campaign_id:
                    sub_campaigns.append(bidding_environment.get_subcampaign(i).copy(new_id=new_sub_campaign_id))
                    new_sub_campaign_id += 1
            temp_environment = BiddingEnvironment(sub_campaigns)
        all_disaggregations.append(temp_environment)
        
    # ora che ho tutte le possibili modalità di disaggregare la subcampaign 0, devo andare a trainare i GP con i sample
    # che ho raccolto nei giorni precedenti (considerando che so a quale curva appartengono grazie al mio algoritmo di
    # disaggregazione). Una volta trainati con questi valori posso recuperare dei sample e vedere se performa meglio la 
    # subcampaign disaggregata o quella aggregata (in qualunque modo lo sia).

    best_disaggregated_environment = None
    best_disaggregated_combinatorial_learner = None
    best_total_reward = -np.inf

    for disaggregated_environment in all_disaggregations:
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
            # print('Temp best: num of subcampaigns -> ', disaggregated_environment.num_subcampaigns, ' total reward -> ', disaggregation_total_reward)
            best_disaggregated_environment = disaggregated_environment.copy()
            best_disaggregated_combinatorial_learner = disaggregated_combinatorial_learner
        
    print('Final best ==> # Subcampaigns ', best_disaggregated_environment.num_subcampaigns, '  -  Total reward ', best_total_reward)
    
    return best_disaggregated_combinatorial_learner, best_disaggregated_environment




def partition(collection):
    if len(collection) == 1:
        yield [ collection ]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert first in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put first in its own subset 
        yield [ [ first ] ] + smaller



"""
        funcs_indexes = list(self.get_functions_ids())
        possible_partitions = create_partitions(funcs_indexes)
        sub_campaigns = []
        for pp in possible_partitions:
            new_sub_campaign_id = 1
            sub_sub_campaigns = []
            for i in pp:
                functions = []
                already_collected_samples = []
                already_pulled_arms = []
                for j in i:
                    functions.append(self.get_function(j))
                    already_collected_samples.append(self.get_function_collected_samples(j).copy())
                    already_pulled_arms = self.pulled_arms.copy()
                sub_sub_campaigns.append(Subcampaign(functions, sub_campaign_id=new_sub_campaign_id, func_numbers=tuple(i),
                        already_observed_samples=already_collected_samples, already_pulled_arms=already_pulled_arms))
                new_sub_campaign_id += 1
            sub_campaigns.append(sub_sub_campaigns)
        return sub_campaigns
        """