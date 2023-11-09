from itertools import count

import matplotlib.pyplot as plt
import torch

from agents.dqn_agent import DQNAgent
from main import BlackJackUltimate


def plot_reward(show_result=False):
    plt.figure(1)
    reward = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    # Take 100 episode averages and plot them too
    if len(reward) >= 100:
        means = reward.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), label="Means Reward")

    plt.pause(0.001)  # pause a bit so that plots are updated


def get_next_state(done, observation):
    if done:
        return None
    else:
        # return torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        return observation


def train_agent(agent, current_state, action_sample, device):
    action = agent.select_action(current_state, action_sample)
    observation, reward, terminated = env.external_step(action.item())

    # smells bad but does not care tbh
    training_states.append([i_episode, observation.tolist(), action.item(), reward])

    reward = torch.tensor([reward], device=device)
    current_done = terminated
    current_next_state = get_next_state(current_done, observation)
    agent.learn(current_state, action, current_next_state, reward)
    current_state = current_next_state

    return current_done, current_state, reward


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
episode_rewards = []
episode_rewards_o = []

wins_x_count = 0
wins_o_count = 0
draws_count = 0

wins_x = []
wins_o = []
draws = []

if torch.cuda.is_available():
    num_episodes = 10000
else:
    num_episodes = 200

training_states = []

env = BlackJackUltimate(is_external=True)

state = env.reset()
agent = DQNAgent(device, state, env.get_action_num())
# agent_x = RandomAgent(env.get_action_num())

for i_episode in range(num_episodes + 1):
    state = env.reset()
    done = False

    for t in count():
        done, state, reward = train_agent(agent, state, env.get_action_sample(), device)

        if done:
            episode_rewards.append(reward)
            plot_reward()
            break

    # if i_episode % 100 == 0:
    #     torch.save(agent.policy_net.state_dict(), "./weights/agent_x_policy.pt")
    #     torch.save(agent.target_net.state_dict(), "./weights/agent_x_target.pt")

# replay_manager = ReplayManager()
# replay_manager.save_to_file(training_states)

# plot_reward(show_result=True)
plt.ioff()
plt.show()
