import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from typing import Any

################################### Q-learning algorithm #################################
def Q_Learning(num_episodes, learning_rate, discount_factor, epsilon, render):

    # Initialize Frozen Lake Envrionment
    env1 = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode='human' if render else None)

    # Initialize Q Table to fit 8x8 size of environment
    q = np.zeros((env1.observation_space.n, env1.action_space.n))

    # Decay rate of epsilon, minimum of 10,000 episodes as well as Random Number Generator
    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()

    QL_rewards = []
    rewards_per_attempt = np.zeros(num_episodes)
    sum_of_rewards_QL = np.zeros(num_episodes)
    count_reward_QL = 0

    #For Loop to run based on input number of episodes. 
    #Q-Learning Algorithm is implemented based on the reset state at start of each loop.
    for p in range(num_episodes):
        state = env1.reset()[0]
        total_reward = 0
        terminated = False
        truncated = False

        # Run environment until 200 moves or until fall through ice.
        while (not terminated and not truncated):
            # Decision to Explore (epsilon) or Exploit
            if rng.random() < epsilon:
                action = env1.action_space.sample()
            else:
                action = np.argmax(q[state,:])

            next_state, reward, terminated, truncated, _ = env1.step(action)

            # Q-Learning Algorithm
            q[state, action] =  q[state, action] + learning_rate * (reward + discount_factor * np.max(q[next_state,:]) - q[state, action])

            state = next_state
            total_reward = total_reward + reward

        # Decay Epsilon so agent begins to transition from Explore to Exploit
        epsilon = max(epsilon - epsilon_decay_rate, 0)

        # Full exploit after epsilon has decayed completely. 
        if(epsilon==0):
            learning_rate = 0.0001

        # Creation of arrays for graphical purposes
        if reward == 1:
            rewards_per_attempt[p] = 1
            count_reward_QL = count_reward_QL + 1
            sum_of_rewards_QL[p] = count_reward_QL
        else:
            sum_of_rewards_QL[p] = count_reward_QL

        # Visual print out to show algorithm is running
        print("count reward_QL: ", count_reward_QL, 'position: ', p)

        QL_rewards.append(total_reward)

    env1.close()

    total_rewards = 0
    num_tests = 5
    # Visualize Q-Learning Policy
    env1 = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode = 'human')
    for x in range(num_tests):
        state = env1.reset()[0]
        terminated = False
        truncated = False
        while (not terminated and not truncated):
            action = np.argmax(q[state,:])
            state, reward, terminated, truncated, _ = env1.step(action)
            total_rewards += reward
    env1.close()

# ## PLOT OF CUMULATIVE REWARDS FOR COMPARISON TO OTHER ALGORITHMS ###
#     plt.plot(sum_of_rewards_QL, label = "Q Learning")
#     plt.xlabel('Attempt')
#     plt.ylabel("Rewards")
#     plt.title('Total Rewards in Frozen Lake Env: Monte Carlo vs Q-Learning')
#     plt.legend()
#     plt.savefig('frozen_lake_Comparison_total_Rewards.png')



# ### PLOT OF REWARDS / 100 ACTIONS FOR COMPARISON TO OTHER ALGORITHMS ###
#     graph=[]
#     sum_rewards_QL = np.zeros(num_episodes)
#     #Calculate average rewards over 100 attempts
#     for t in range(num_episodes):
#         sum_rewards_QL[t] = np.sum(rewards_per_attempt[max(0, t-100):(t+1)])
#         graph.append(sum_rewards_QL[t])
    
#     plt.plot(sum_rewards_QL, label = "Q Learning")
#     plt.xlabel('Attempt')
#     plt.ylabel("Rewards / 100 Attempts")
#     plt.title('Frozen Lake MC-Learning vs Q-Learning')
#     plt.legend()
#     plt.savefig('frozen_lake_rewards_100_attempts.png')

    return QL_rewards, count_reward_QL


##################### MONTE CARLO ALGORITHM ############################
def MC_Learning(num_episodes, gamma, epsilon):

    # Initialize Frozen Lake Envrionment
    env3 = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False)

    # Initialize Q Table and state action returns table to fit 8x8 size of environment   
    Q = np.zeros((env3.observation_space.n, env3.action_space.n))
    returns = {(s, a): [0] for s in range(env3.observation_space.n) for a in range(env3.action_space.n)}

    # Decay rate of epsilon, minimum of 10,000 episodes as well as Random Number Generator
    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng() 
    cumulative_rewards = []
    rewards_per_attempt_MC = np.zeros(num_episodes)
    sum_of_rewards_MC = np.zeros(num_episodes)
    count_reward_MC = 0

    #For Loop to run based on input number of episodes. 
    #Monte Carlo Algorithm is implemented based on the reset state at start of each loop.
    for i in range(num_episodes):
        state = env3.reset()[0]
        terminated = False
        truncated = False
        episode = []
        total_reward = 0

        # Run environment until 200 moves or until fall through ice.
        while (not terminated and not truncated):
            # Decision to Explore (epsilon) or Exploit
            if rng.random() < epsilon:
                action = env3.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            next_state, reward, terminated , truncated, _ = env3.step(action)
            episode.append((state, action, reward))
            total_reward = total_reward + reward
            state = next_state

         # Decay Epsilon so agent begins to transition from Explore to Exploit       
        epsilon = max(epsilon - epsilon_decay_rate, 0)

        # Creation of arrays for graphical purposes
        if reward == 1:
            rewards_per_attempt_MC[i] = 1
            count_reward_MC = count_reward_MC + 1
            sum_of_rewards_MC[i] = count_reward_MC
        else:
            sum_of_rewards_MC[i] = count_reward_MC

        # Visual print out to show algorithm is running
        print("count reward_MC: ", count_reward_MC, "position: ", i)

        cumulative_rewards.append(total_reward)


        states, actions, rewards = zip(*episode)
        discounts = np.array([gamma**i for i in range(len(rewards)+1)])

        # Monte Carlo Algorithm, G=rewards discounted by input gamma
        for z, (state, action) in enumerate(zip(states, actions)):
            G = sum(rewards[z:] * discounts[:-(1+z)])
            returns[(state, action)].append(G)
            Q[state, action] = np.mean(returns[(state, action)])

    env3.close()

    ### IF WE WANT TO VISUALIZE THE POLICY ###
    total_rewards = 0
    num_tests = 5
    # Visualize Monte Carlo Policy
    env3 = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode = 'human')
    for x in range(num_tests):
        state = env3.reset()[0]
        terminated = False
        truncated = False
        while (not terminated and not truncated):
            action = np.argmax(Q[state,:])
            state, reward, terminated, truncated, _ = env3.step(action)
            total_rewards += reward
    env3.close()

# ### GRAPH OF MONTE CARLO FOR LEARNING PURPOSES ###
#     graph_MC = []
#     sum_rewards_MC = np.zeros(num_episodes)
#      #Calculate average rewards over 100 attempts
#     for n in range(num_episodes):
#         sum_rewards_MC[n] = np.sum(rewards_per_attempt_MC[max(0, n-100):(n+1)])
#         graph_MC.append(sum_rewards_MC[n])

#     plt.plot(sum_rewards_MC, label = "Monte Carlo Learning")
#     plt.xlabel('Attempt')
#     plt.ylabel("Rewards / 100 Attempts")
#     plt.title('Frozen Lake MC-Learning vs Q-Learning')
#     plt.legend()
#     plt.savefig('frozen_lake_rewards_100_attempts.png')
#     plt.close()


# ### PLOT OF CUMULATIVE REWARDS FOR COMPARISON TO OTHER ALGORITHMS ###
#     plt.plot(sum_of_rewards_MC, label = "Monte Carlo Learning")
#     plt.xlabel('Attempt')
#     plt.ylabel("Rewards")
#     plt.title('Total Rewards in Frozen Lake Env: Monte Carlo vs Q-Learning')
#     plt.legend()
#     plt.savefig('frozen_lake_Comparison_total_Rewards.png')
#     plt.close()

    return cumulative_rewards, count_reward_MC

### MAIN FUNCTION ###
def main():
    ### RUN EITHER FOR RL METHOD ###
    QL_result = Q_Learning(num_episodes = 12000, learning_rate = 0.1, discount_factor=0.9, epsilon=1.0, render = False)
    MC_result = MC_Learning(num_episodes = 12000, gamma = 0.8, epsilon = 1.0)


# ## Q Learning Simulation AND Finding Hyper Parameters ###
#     QL_total_reward = 'QL_total_alpha5.csv'
#     QL_all_env_rewards = 'QL_environment_rewards5.csv'
#     alpha = [0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4]
#     gamma = [0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]

#     for a, g in zip(alpha, gamma):
#         hyperparameters_QL = []
#         all_cumulative_rewards = []

#     QL_rewards = []
#     QL_runtime_array = []

#     for i in range (30):
#         start_time_QL: float = time.time()
#         QL_result = Q_Learning(num_episodes = 12000, learning_rate = 0.1, discount_factor=0.9, epsilon=1.0, is_training=True, render = False)
#         end_time_QL: float = time.time()
#         QL_runtime = end_time_QL - start_time_QL
#         QL_rewards.append(QL_result[1])
#         QL_runtime_array.append(QL_runtime)

#     df_QL = pd.DataFrame(QL_rewards)
#     df_QL.to_csv('QL_rewards_for_average.csv', index=False, header=False, mode='a')

#     df_QL_runtime = pd.DataFrame(QL_runtime_array)
#     df_QL_runtime.to_csv('QL_runtime_for_average.csv', index=False, header=False, mode='a')


#     hyperparameters_QL.append({'alpha/gamma':f'{a},{g}', 'Total Rewards':QL_result[1]}) #Total Rewards from env
#     all_cumulative_rewards.append({'alpha/gamma':f'{a},{g}', 'Total Event Rewards': QL_result[0]})

        
#     df_QL = pd.DataFrame(hyperparameters_QL)

#     df_QL = pd.DataFrame(hyperparameters_QL)
#     df_QL.to_csv(QL_total_reward, index=False, header=False, mode='a')

#     df_QL_rewards = pd.DataFrame(all_cumulative_rewards)

#     df_QL_rewards = pd.DataFrame(all_cumulative_rewards)
#     df_QL_rewards.to_csv(QL_all_env_rewards, index=False, header=False, mode='a')

# ### PLOT FOR Q LEARNING HP EVALUATION ###
#     df = pd.read_csv('QL_environment_rewards.csv', header=0)
#     column_titles = df.columns
#     data = df.values

#     for i in range(len(column_titles)):
#         running_sum = [sum(data[:j+1, i]) for j in range(len(data[:, i]))]
#         plt.plot(running_sum, label=f'Gamma: {column_titles[i]}')

#     plt.title('Total Rewards in Frozen Lake Env: Q-Learning Hyperparameters')
#     plt.xlabel('Attempts')
#     plt.ylabel('Cumulative Rewards')
#     plt.legend()
#     plt.savefig('frozen_lake_QL_HP_comparison.png')
#     plt.close()


# ## Monte Carlo Simulation AND Finding Hyper Parameters ###
#     MC_total_reward = 'MC_total_gamma_08.csv'
#     MC_all_env_rewards = 'MC_environment_rewards_gamma_08.csv'
#     hyperparameters_MC = []
#     all_cumulative_rewards = []
#     MC_rewards = []
#     MC_runtime_array = []

#     for i in range(10):    
#         start_time_MC: float = time.time()
#         MC_result = MC_Learning(num_episodes = 12000, gamma = 0.8, epsilon = 1.0)
#         end_time_MC: float = time.time()
#         MC_runtime = end_time_MC - start_time_MC
#         MC_rewards.append(MC_result[1])
#         MC_runtime_array.append(MC_runtime)

#     df_MC = pd.DataFrame(MC_rewards)
#     df_MC.to_csv('MC_rewards_for_average.csv', index=False, header=False, mode='a')

#     df_MC_runtime = pd.DataFrame(MC_runtime_array)
#     df_MC_runtime.to_csv('MC_runtime_for_average.csv', index=False, header=False, mode='a')

#     hyperparameters_MC.append({'gamma':0.8, 'Total Rewards':MC_result[1]}) #Total Rewards from env
#     all_cumulative_rewards.append({'gamma': 0.8, 'Total Event Rewards': MC_result[0]})

#     print('\n', hyperparameters_MC)
    
#     df_MC = pd.DataFrame(hyperparameters_MC)

#     df_MC = pd.DataFrame(hyperparameters_MC)
#     df_MC.to_csv(MC_total_reward, index=False, header=False, mode='a')

#     # Write all_cumulative_rewards to Excel
#     df_MC_rewards = pd.DataFrame(all_cumulative_rewards)

#     df_MC_rewards = pd.DataFrame(all_cumulative_rewards)
#     df_MC_rewards.to_csv(MC_all_env_rewards, index=False, header=False, mode='a')


# ### PLOT FOR MONTE CARLO GAMMA EVALUATION ###
#     df = pd.read_csv('MC_environment_rewards.csv', header=0)  # Assuming the first row contains column headers

#     column_titles = df.columns
#     data = df.values

#     for i in range(len(column_titles)):
#         running_sum = [sum(data[:j+1, i]) for j in range(len(data[:, i]))]
#         plt.plot(running_sum, label=f'Gamma: {column_titles[i]}')

#     plt.title('Total Rewards in Frozen Lake Env: Monte Carlo Gamma Optimization')
#     plt.xlabel('Attempts')
#     plt.ylabel('Cumulative Rewards')
#     plt.legend()
#     plt.savefig('frozen_lake_MC_gamma_comparison.png')
#     plt.close()

if __name__ == "__main__":
    main()