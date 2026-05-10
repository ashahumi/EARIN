import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def get_discrete_state(state, env, bins):
    """
    Quantizes the continuous observation space into discrete bins for the Q-table.
    """
    bin_size = (env.observation_space.high - env.observation_space.low) / bins
    discrete_state = (state - env.observation_space.low) / bin_size
    return tuple(discrete_state.astype(int))

def train_q_learning(learning_rate, discount_factor, epsilon_decay, episodes=20000):
    """
    Initializes and trains the Q-learning algorithm on the MountainCar environment.
    """
    # Initialization of the environment
    env = gym.make("MountainCar-v0")
    
    # Initialization of Q-table and hyperparameters
    bins = [20, 20] # 20 bins for position, 20 for velocity
    q_table = np.random.uniform(low=-2, high=0, size=(bins + [env.action_space.n]))
    
    epsilon = 1.0
    min_epsilon = 0.01
    
    # Metrics logging
    rewards_per_episode = []
    
    # Training Loop
    for episode in range(episodes):
        state, _ = env.reset()
        discrete_state = get_discrete_state(state, env, bins)
        
        done = False
        total_reward = 0
        
        while not done:
            # Choose an action (Epsilon-Greedy policy)
            if np.random.random() > epsilon:
                action = np.argmax(q_table[discrete_state]) # Exploit
            else:
                action = env.action_space.sample() # Explore
                
            # Perform action and observe state and reward
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            new_discrete_state = get_discrete_state(new_state, env, bins)
            
            # Update the Q-table
            if not done:
                max_future_q = np.max(q_table[new_discrete_state])
                current_q = q_table[discrete_state + (action,)]
                
                # Q-learning equation
                new_q = current_q + learning_rate * (reward + discount_factor * max_future_q - current_q)
                q_table[discrete_state + (action,)] = new_q
                
            elif new_state[0] >= env.unwrapped.goal_position: # Reached the flag
                q_table[discrete_state + (action,)] = 0 
                
            discrete_state = new_discrete_state
            total_reward += reward
            
        # Decay epsilon
        if epsilon > min_epsilon:
            epsilon *= epsilon_decay
            
        rewards_per_episode.append(total_reward)
            
    env.close()
    return rewards_per_episode, q_table

def run_comparison_experiments():
    """
    Runs training loops with varying hyperparameters and plots the comparisons.
    """
    # Base hyperparameters to use when isolating one variable
    base_lr = 0.1
    base_gamma = 0.99
    base_decay = 0.95
    episodes = 20000
    window = 100 # Window for calculating the moving average

    # --- 1. Compare Learning Rates ---
    lr_values = [0.05, 0.1, 0.5, 0.8]
    plt.figure(figsize=(12, 6))
    for lr in lr_values:
        print(f"Testing Learning Rate: {lr}...")
        rewards, _ = train_q_learning(learning_rate=lr, discount_factor=base_gamma, epsilon_decay=base_decay, episodes=episodes)
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(moving_avg, label=f"LR = {lr}")
    
    plt.title("Impact of Learning Rate on MountainCar Performance")
    plt.xlabel(f"Episode (Moving Average over {window})")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- 2. Compare Discount Factors ---
    gamma_values = [0.8, 0.95, 0.99, 0.999]
    plt.figure(figsize=(12, 6))
    for gamma in gamma_values:
        print(f"Testing Discount Factor: {gamma}...")
        rewards, _ = train_q_learning(learning_rate=base_lr, discount_factor=gamma, epsilon_decay=base_decay, episodes=episodes)
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(moving_avg, label=f"Gamma = {gamma}")
    
    plt.title("Impact of Discount Factor on MountainCar Performance")
    plt.xlabel(f"Episode (Moving Average over {window})")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- 3. Compare Epsilon Decay ---
    decay_values = [0.8, 0.95, 0.99, 0.999]
    plt.figure(figsize=(12, 6))
    for decay in decay_values:
        print(f"Testing Epsilon Decay: {decay}...")
        rewards, _ = train_q_learning(learning_rate=base_lr, discount_factor=base_gamma, epsilon_decay=decay, episodes=episodes)
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(moving_avg, label=f"Decay = {decay}")
    
    plt.title("Impact of Epsilon Decay on MountainCar Performance")
    plt.xlabel(f"Episode (Moving Average over {window})")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    print("Starting Q-Learning Hyperparameter Comparisons (5000 Episodes)...")
    run_comparison_experiments()