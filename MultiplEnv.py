import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import tqdm
import matplotlib.pyplot as plt

# VISUALIZING THE ACTION DISTRIBUTION

def visualize(policy, continuous):

    state = torch.FloatTensor(np.random.randn(state_dim))  # Random állapot

    if continuous:
        mean, std = policy(state)
        dist = torch.distributions.Normal(mean.detach().squeeze(), std.detach().squeeze())
        samples = dist.sample((1000,))  # Mintavételezünk az eloszlásból
        plt.hist(samples.numpy(), bins=30, density=True)
        plt.title("Continuous Action Distribution (Gaussian)")
        plt.xlabel("Action value")
        plt.ylabel("Probability Density")
    else:
        probs = policy(state).detach().numpy()
        plt.bar(range(len(probs)), probs)
        plt.title("Discrete Action Probabilities (Softmax)")
        plt.xlabel("Action index")
        plt.ylabel("Probability")

    plt.show()

# 1. POLICY NETWORK

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, continuous=False, contiuous=None):
        super(PolicyNetwork, self).__init__()
        self.continuous = continuous
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2 * action_dim if continuous else action_dim),
        )

    #The constructor method initializes the neural network with two fully connected layers. The first layer takes the input dimension (actual_state_dim) and maps it to 16 hidden units using a linear transformation followed by a ReLU activation function.
    #The second layer takes the 16 hidden units and maps them to the output dimension (actual_action_dim) using another linear transformation.

    def forward(self, state):
        out = self.fc(state)
        if self.continuous:
            mean, log_std = torch.chunk(out, 2, dim=-1) # Split the output into mean and log_std
            std = torch.exp(log_std) # Deviation needs to be positive
            return mean, std
        else:
            return torch.softmax(out,dim=-1) # In case of discrete action space we use softmax

    #The forward method is responsible for defining the forward pass of the network, which is how the input data flows through the network layers to produce an output.
    #In summary, the forward method takes an input state, processes it through the network's layers, and returns the resulting action probabilities.

# 2. GAME AND LEARNING
while True:
    print('"1" CartPole-v1 (discrete)')
    print('"2" Pendulum-v1 (continuous)')
    print('"x" Exit')
    choice = input('Which environment would you like to choose? ')

    if choice == '1':
        print('"1" New')
        print('"2" Fine-Tune')
        print('"x" Back')
        option = input('Which option would you like to choose? ')

        if option == '1':
            env = gym.make("CartPole-v1")
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            continuous = False
        elif option == '2':
            env = gym.make("CartPole-v1")
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            continuous = False
            policy.load_state_dict(torch.load('multiplenv_policy_weights.pth'))
        elif option == 'x':
            continue
        else:
            print('Invalid choice')
    elif choice == '2':
        print('"1" New')
        print('"2" Fine-Tune')
        print('"x" Back')
        option = input('Which option would you like to choose? ')

        if option == '1':
            env = gym.make("Pendulum-v1")
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            continuous = True
        elif option == '2':
            env = gym.make("Pendulum-v1")
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            continuous = True
            policy.load_state_dict(torch.load('multiplenv_policy_weights.pth'))
        elif option == 'x':
            continue
        else:
            print('Invalid choice')
    elif choice == 'x':
        exit()
    else:
        print('Invalid choice')

    policy = PolicyNetwork(state_dim, action_dim, continuous)
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    episode_rewards = []

    # 3. CHOOSING AN ACTION

    def select_action(state):
        state = torch.FloatTensor(state) #The function converts the input state, which is typically a NumPy array, into a PyTorch tensor
        if continuous:
            mean, std = policy(state)
            dist = torch.distributions.Normal(mean, std) # Gaussian policy
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            return action.detach().numpy(), log_prob
        else:
            probs = policy.forward(state) #the function passes the state tensor through the policy network to obtain the action probabilities
            action = torch.multinomial(probs, 1).item() #returns the index of the selected action, which is then converted to a Python integer
            return action, torch.log(probs[action])

    # 4. POLICY UPDATE

    def update_policy(episode_log_probs, rewards):
        G = 0 #discounted return
        loss = 0 # total loss of the episode
        for log_prob, reward in zip(episode_log_probs, reversed(rewards)):
            G = reward + G * 0.99 # discount factor
            loss -= log_prob * G
        optimizer.zero_grad() # resets gradients to zero
        loss.backward() # compute gradients of the loss
        optimizer.step() # update network parameters

    # 5. TRAINING LOOP

    for episode in tqdm.tqdm(range(100)):
        state = env.reset()[0]
        episode_log_probs = []
        rewards = []

        for _ in range(200):
            action, log_prob = select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            episode_log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
            if done:
                break

        episode_rewards.append(sum(rewards))  # Az epizód összesített jutalma
        update_policy(episode_log_probs, rewards)

    env.close()
    visualize(policy, continuous)
    print(f"Training of option {choice} completed!")

    # SAVING THE WEIGHTS

    torch.save(policy.state_dict(), 'multiplenv_policy_weights.pth')
    print("Model weights saved!")