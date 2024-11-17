import random
import csv
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

# Rewards for each bandit
Bandit_Reward = [1, 2, 3, 4]

class Bandit(ABC):
    @abstractmethod
    def __init__(self, p, reward):
        self.p = p  # True probability of winning 
        self.n_trials = 0
        self.total_reward = 0
        self.reward = reward  #  reward associated with this bandit

    @abstractmethod
    def __repr__(self):
        return f"Bandit with true probability {self.p} and reward {self.reward}"

    @abstractmethod
    def pull(self):
        return self.reward if random.random() < self.p else 0

    @abstractmethod
    def update(self, reward):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        pass


# Epsilon-Greedy Strategy
class EpsilonGreedy(Bandit):
    def __init__(self, p, reward, epsilon=0.1):
        super().__init__(p, reward)
        self.epsilon = epsilon
        self.est_mean = 0

    def __repr__(self):
        return f"EpsilonGreedy Bandit with true probability {self.p} and reward {self.reward}"

    def pull(self):
        if random.random() < self.epsilon:
            return random.choice(Bandit_Reward)  # Random action among bandit rewards
        else:
            return self.reward if random.random() < self.est_mean else 0

    def update(self, reward):
        self.n_trials += 1
        self.est_mean += (reward - self.est_mean) / self.n_trials

    def experiment(self, n_trials):
        rewards = []
        for t in range(1, n_trials + 1):
            self.epsilon = 1 / t  # Decaying epsilon
            reward = self.pull()
            self.update(reward)
            self.total_reward += reward
            rewards.append(reward)
        return rewards

    def report(self):
        avg_reward = self.total_reward / self.n_trials
        regret = self.n_trials * max(Bandit_Reward) - self.total_reward
        print(f"Algorithm: {self.__class__.__name__}")
        print(f"Total Reward: {self.total_reward:.2f}")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Cumulative Regret: {regret:.2f}")
        logger.info(f"EpsilonGreedy Bandit: Total Reward = {self.total_reward}, Avg Reward = {avg_reward:.3f}, Cumulative Regret = {regret}")


# Thompson Sampling Strategy
class ThompsonSampling(Bandit):
    def __init__(self, p, reward):
        super().__init__(p, reward)
        self.a = 1
        self.b = 1

    def __repr__(self):
        return f"ThompsonSampling Bandit with true probability {self.p} and reward {self.reward}"

    def pull(self):
        sample = np.random.beta(self.a, self.b)
        return self.reward if random.random() < sample else 0

    def update(self, reward):
        self.n_trials += 1
        if reward > 0:
            self.a += 1
        else:
            self.b += 1

    def experiment(self, n_trials):
        rewards = []
        for _ in range(n_trials):
            reward = self.pull()
            self.update(reward)
            self.total_reward += reward
            rewards.append(reward)
        return rewards

    def report(self):
        avg_reward = self.total_reward / self.n_trials
        regret = self.n_trials * max(Bandit_Reward) - self.total_reward
        print(f"Algorithm: {self.__class__.__name__}")
        print(f"Total Reward: {self.total_reward:.2f}")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Cumulative Regret: {regret:.2f}")
        logger.info(f"ThompsonSampling Bandit: Total Reward = {self.total_reward}, Avg Reward = {avg_reward:.3f}, Cumulative Regret = {regret}")

class Visualization:
    @staticmethod
    def plot1(rewards_eg, rewards_ts):
        plt.figure(figsize=(12, 5))
        plt.plot(np.cumsum(rewards_eg), label='Epsilon-Greedy Cumulative Reward')
        plt.plot(np.cumsum(rewards_ts), label='Thompson Sampling Cumulative Reward')
        plt.xlabel('Trial')
        plt.ylabel('Cumulative Reward')
        plt.title('Learning Process for Epsilon-Greedy and Thompson Sampling')
        plt.legend()
        plt.show()

    @staticmethod
    def plot2(rewards_eg, rewards_ts):
        plt.figure(figsize=(12, 5))
        plt.plot(np.cumsum(rewards_eg), label='Epsilon-Greedy Cumulative Reward')
        plt.plot(np.cumsum(rewards_ts), label='Thompson Sampling Cumulative Reward')
        plt.xlabel('Trial')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Rewards Comparison')
        plt.legend()
        plt.show()


# Experimentation and Reporting
def save_rewards_to_csv(data):
    with open('bandit_reward.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Bandit', 'Reward', 'Algorithm'])
        writer.writerows(data)

def comparison():
    n_trials = 20000
    bandit_probs = [0.1, 0.2, 0.3, 0.4]  # Example probabilities for bandits
    all_rewards_data = []  # This will collect data for saving

    # Run Epsilon-Greedy with fixed rewards
    eg_bandits = [EpsilonGreedy(p, reward) for p, reward in zip(bandit_probs, Bandit_Reward)]
    eg_rewards = [bandit.experiment(n_trials) for bandit in eg_bandits]

    # Collect data for CSV
    for i, rewards in enumerate(eg_rewards):
        for reward in rewards:
            all_rewards_data.append([i, reward, "Epsilon-Greedy"])

    # Run Thompson Sampling with fixed rewards
    ts_bandits = [ThompsonSampling(p, reward) for p, reward in zip(bandit_probs, Bandit_Reward)]
    ts_rewards = [bandit.experiment(n_trials) for bandit in ts_bandits]

    # Collect data for CSV
    for i, rewards in enumerate(ts_rewards):
        for reward in rewards:
            all_rewards_data.append([i, reward, "Thompson Sampling"])

    # Visualize Results
    avg_rewards_eg = [sum(rewards) for rewards in eg_rewards]
    avg_rewards_ts = [sum(rewards) for rewards in ts_rewards]
    Visualization.plot1(avg_rewards_eg, avg_rewards_ts)
    Visualization.plot2(avg_rewards_eg, avg_rewards_ts)

    # Save data to CSV
    save_rewards_to_csv(all_rewards_data)


if __name__ == '__main__':
    comparison()
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

