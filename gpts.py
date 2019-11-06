from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np 
import matplotlib.pyplot as plt






class CombinatorialGPTSLearner():

    def __init__(self, gpts_subproblems):
        self.gpts_subproblems = gpts_subproblems

    def get_gpts_subproblem(self, gpts_index):
        for gp_sub in self.gpts_subproblems:
            if gp_sub.get_gp_id() == gpts_index:
                return gp_sub
        return None

    def predict(self):
        samples = []
        for gp in self.gpts_subproblems:
            prediction = gp.get_prediction()
            if sum(prediction) == 0:
                prediction = [i * 1e-3 for i in range(gp.num_arms)]
            prediction[0] = 0.0
            samples.append(prediction)
        return samples
    
    def observe_and_update(self, super_arm, bidding_environment):
        total_reward = 0
        for (sc_id, arm_bid) in super_arm:
            reward = bidding_environment.get_subcampaign(sc_id).sample(arm_bid)
            # print('reward: ', reward, '   arm_bid: ', arm_bid, '   total rew: ', total_reward)
            # TODO: do I pass the total reward or the reward of each function?
            self.get_gpts_subproblem(sc_id).update(arm_bid, reward)
            total_reward += sum(reward)
        return total_reward
    
    def get_total_reward(self, super_arm, bidding_environment):
        total_reward = 0
        for (sc_id, arm_bid) in super_arm:
            reward = bidding_environment.get_subcampaign(sc_id).sample(arm_bid)
            total_reward += sum(reward)
        return total_reward

    # TODO: revisit this method
    def copy(self):
        return CombinatorialGPTSLearner([gp.copy() for gp in self.gpts_subproblems])








class GPTSLearner():
    def __init__(self, num_arms, arms_bids, gp_id):
        self.gp_id = gp_id
        self.num_arms = num_arms
        self.arms_bids = arms_bids
        self.predicted_values = np.zeros(self.num_arms)
        self.means = np.zeros(self.num_arms)
        self.sigmas = np.ones(self.num_arms) * 10
        self.pulled_arms = []
        self.collected_rewards = np.array([])
        self.rewards_per_arm = [[] for _ in range(self.num_arms)]
        alpha = 10.0
        kernel = C(1.0, (1e-3,1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel = kernel, alpha = alpha**2, normalize_y = True, n_restarts_optimizer = 9)

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x,y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms_bids).T, return_std=True)
        # self.predicted_values = np.random.normal(self.means, self.sigmas)
        self.predicted_values = self.means.copy()

    def update_observations(self, pulled_arm, reward):
        arm_idx = self.get_arm_index(pulled_arm)
        self.rewards_per_arm[arm_idx].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
        self.pulled_arms.append(pulled_arm)

    def update(self, pulled_arm, reward):
        num_rewards = len(reward)
        for rew in reward:
            self.update_observations(pulled_arm, rew * num_rewards)
        # total_reward = sum(reward)
        # self.update_observations(pulled_arm, total_reward)
        self.update_model()

    def get_arm_index(self, pulled_arm):
        for i in range(self.num_arms):
            if self.arms_bids[i] == pulled_arm:
                return i 
        return False

    def get_prediction(self):
        return self.predicted_values

    def get_arm_prediction(self, arm):
        arm_idx = self.get_arm_index(arm)
        return self.predicted_values[arm_idx]

    def get_arm_std(self, arm):
        arm_idx = self.get_arm_index(arm)
        return self.sigmas[arm_idx]
    
    def get_gp_id(self):
        return self.gp_id 

    def regression_error(self):
        return

    # TODO: revisit this method
    def copy(self):
        gp = GPTSLearner(self.num_arms, self.arms_bids, self.gp_id)
        gp.predicted_values = self.predicted_values.copy()
        gp.pulled_arms = self.pulled_arms.copy()
        gp.rewards_per_arm = self.rewards_per_arm.copy()
        gp.collected_rewards = self.collected_rewards.copy()

    def plot(self, dis=False):
        means, _ = self.gp.predict(np.atleast_2d(self.arms_bids).T, return_std = True)
        plt.figure()
        plt.plot(self.arms_bids, means)
        if dis:
            plot_path = '/Users/alberto/development/dia/advertising/plots/GPTS_disag_' + str(self.get_gp_id()) + '.png'
        else:
            plot_path = '/Users/alberto/development/dia/advertising/plots/GPTS_' + str(self.get_gp_id()) + '.png'
        plt.savefig(plot_path, bbox_inches='tight', dpi = 300)
        plt.close()