"""
Baseline of SAC taken from: https://github.com/vaishak2future/sac/blob/master/sac.ipynb
Autor: Amir Belder
"""

import datetime
import math
import os
import random
import tempfile
from datetime import date
from pathlib import Path

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from ba_env import BaEnv
from torch.distributions import Normal

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class ReplayBuffer:
    def __init__(self, capacity, random_chosing=False):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.random_chosing = random_chosing

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if self.random_chosing is False:
            batch = self.buffer[-batch_size:]
        else:
            batch = random.sample(self.buffer, batch_size)

        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_actions,
        hidden_size,
        init_w=3e-3,
        log_std_min=-20,
        log_std_max=2,
    ):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z.to(device))
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(
            1 - action.pow(2) + epsilon
        )
        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample().to(device)
        action = torch.tanh(mean + std * z)

        action = action.cpu()  # .detach().cpu().numpy()
        return action[0]


def update(
    batch_size,
    gamma=0.99,
    soft_tau=1e-2,
):
    global replay_buffer

    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action = torch.FloatTensor(action).to(device)
    reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    predicted_q_value1 = soft_q_net1(state, action)
    predicted_q_value2 = soft_q_net2(state, action)
    predicted_value = value_net(state)
    new_action, log_prob, epsilon, mean, log_std = policy_net.evaluate(state)

    # Training Q Function
    target_value = target_value_net(next_state)
    target_q_value = reward + (1 - done) * gamma * target_value
    q_value_loss1 = soft_q_criterion1(predicted_q_value1, target_q_value.detach())
    q_value_loss2 = soft_q_criterion2(predicted_q_value2, target_q_value.detach())

    soft_q_optimizer1.zero_grad()
    q_value_loss1.backward()
    soft_q_optimizer1.step()
    soft_q_optimizer2.zero_grad()
    q_value_loss2.backward()
    soft_q_optimizer2.step()
    # Training Value Function
    predicted_new_q_value = torch.min(
        soft_q_net1(state, new_action), soft_q_net2(state, new_action)
    )
    target_value_func = predicted_new_q_value - log_prob
    value_loss = value_criterion(predicted_value, target_value_func.detach())

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    # Training Policy Function
    policy_loss = (log_prob - predicted_new_q_value).mean()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    for target_param, param in zip(
        target_value_net.parameters(), value_net.parameters()
    ):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )


env = BaEnv()

action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
hidden_dim = 256

value_net = ValueNetwork(state_dim, hidden_dim).to(device)
target_value_net = ValueNetwork(state_dim, hidden_dim).to(device)

soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)


#Global definitions 

value_criterion = nn.MSELoss()
soft_q_criterion1 = nn.MSELoss()
soft_q_criterion2 = nn.MSELoss()


value_lr = 3e-4
soft_q_lr = 3e-4
policy_lr = 3e-4

value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)
soft_q_optimizer1 = optim.Adam(soft_q_net1.parameters(), lr=soft_q_lr)
soft_q_optimizer2 = optim.Adam(soft_q_net2.parameters(), lr=soft_q_lr)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)


replay_buffer_size = 1000000

# Training Hyperparameters
max_frames = 40000
max_steps = 10000
frame_idx = 0
rewards = []
batch_size = 25  # 128

# In case of a random chosing, a smaller buffer that would hold a few steps back
batch_replay_buffer_size = 20

print(max_steps, batch_size)

last_k_iters_avg = 0
batches_counter = 1
classic_iters = []
convergence_reason_classic = []
step_iters = []
convergence_reason_step = []

learning_start_iters = 500
random_sampling_buffer = True

replay_buffer = None



def main(json_path=None):
    global batch_size
    global batches_counter
    global replay_buffer
    global last_k_iters_avg
    global rewards
    global frame_idx


    json_path = None

    description = "type_experiment_type_here"
    description += (
        "_"
        + str(max_steps)
        + "_max_steps_"
        + str(learning_start_iters)
        + "_learning_start_iters_buffer_random_sampling_is_"
        + str(random_sampling_buffer)
    )


    e = datetime.datetime.now()
    date_str = (
        "_date_"
        + str(e.day)
        + "_"
        + str(e.month)
        + "_hour_"
        + str(e.hour)
        + "_"
        + str(e.minute)
    )
    description += date_str

    tmp_dir = tempfile.mkdtemp()
    result_dir = Path(tmp_dir) / description
    result_dir.mkdir(parents=True, exist_ok=True)

    
        with open(result_dir / "results_sum_up.txt", "w") as f:
            results_dir = str(result_dir)
            if random_sampling_buffer is True:
                batch_size = int(batch_size / batch_replay_buffer_size)

            num_training_experiments = 1
            max_frames = max_steps * num_training_experiments

            # Training phase
            for i in range(num_training_experiments):
                print("training experiment number: ", i)
                lowest_iter_number = np.inf
                replay_buffer = ReplayBuffer(
                    replay_buffer_size, random_chosing=random_sampling_buffer
                )
                if random_sampling_buffer is True:
                    batch_replay_buffer = ReplayBuffer(
                        batch_replay_buffer_size, random_chosing=False
                    )
                    
                # initializing done
                done = False

                # Training Loop
                while frame_idx < max_frames:
                    state = env.reset(iter=i, json_path=json_path)
                    episode_reward = 0
                    if done is True and frame_idx > learning_start_iters:
                        frame_idx = 0
                        break
                    for step in range(max_steps):
                        if frame_idx > learning_start_iters:
                            if frame_idx == learning_start_iters + 1:
                                env.reset(json_path=json_path)
                            env.enable_printing(True)
                            action = policy_net.get_action(state).detach()
                            if action < 0:
                                print("Action out of range")

                            next_state, reward, done, info_dict = env.step(
                                action.numpy()
                            )
                            if next_state is None:
                                print("None next_state")
                            elif reward is None:
                                print("None reward")
                        else:
                            env.enable_printing(False)
                            action = env.action_space.sample()
                            action = action * 1e-1
                            if action < 0 or action > 1:
                                print("Action out of range")
                            next_state, reward, done, info_dict = env.step(action)
                            if next_state is None:
                                print("None next_state")
                            elif reward is None:
                                print("None reward")

                        if str(info_dict["lambda"]).__contains__("nan"):
                            print("bad lambda detected")
                            break
                        if random_sampling_buffer is False:
                            # Insert all states as a continues flow for the network to learn from
                            replay_buffer.push(state, action, reward, next_state, done)
                        else:
                            # Insert small batches that would be randomly chosen
                            if frame_idx % batch_replay_buffer_size == 0:
                                # b_ as in batch
                                if frame_idx == 0:
                                    # first iteration
                                    batch_replay_buffer.push(
                                        state, action, reward, next_state, done
                                    )
                                else:
                                    (
                                        b_state,
                                        b_action,
                                        b_reward,
                                        b_next_state,
                                        b_done,
                                    ) = batch_replay_buffer.sample(
                                        batch_replay_buffer_size
                                    )
                                    replay_buffer.push(
                                        b_state,
                                        b_action,
                                        b_reward,
                                        b_next_state,
                                        b_done,
                                    )
                                    batch_replay_buffer = ReplayBuffer(
                                        batch_replay_buffer_size, random_chosing=False
                                    )
                                    # Add the last iteration ti the new batch_replay_buffer
                                    batch_replay_buffer.push(
                                        state, action, reward, next_state, done
                                    )
                            else:
                                batch_replay_buffer.push(
                                    state, action, reward, next_state, done
                                )
                        if (
                            frame_idx % batch_size == 0
                            and frame_idx < learning_start_iters
                        ):
                            env.reset(json_path=json_path)
                            print("batch number: ", batches_counter)
                            batches_counter += 1

                        if frame_idx % 1000 == 0:
                            print("finished " + str(frame_idx) + " iters")

                        state = next_state
                        episode_reward += reward
                        frame_idx += 1

                        if len(replay_buffer) % max_steps > batch_size:
                            update(batch_size)

                        last_k_iters_avg += episode_reward
                        if frame_idx % 1000 == 0:
                            print(
                                "frame: ",
                                frame_idx,
                                " avg reward of last 1000 iters: ",
                                last_k_iters_avg / 1000,
                            )
                            # plot(frame_idx, rewards)
                            last_k_iters_avg = 0
                        # done = True
                        # if done: #and frame_idx > 1000:
                        if done and frame_idx > learning_start_iters:
                            it = info_dict["iter_num"]
                            step_iters.append(it)
                            convergence_reason_step.append(
                                info_dict["convergence_reason"]
                            )
                            if info_dict["convergence_reason"] == "converged":
                                if it < lowest_iter_number:
                                    lowest_iter_number = it
                                    torch.save(
                                        soft_q_net1.state_dict(),
                                        results_dir + "/soft_q_1.pth",
                                    )
                                    torch.save(
                                        soft_q_net2.state_dict(),
                                        results_dir + "/soft_q_2.pth",
                                    )
                                    torch.save(
                                        policy_net.state_dict(),
                                        results_dir + "/policy.pth",
                                    )
                            break

            rewards.append(episode_reward)

            # testing
            print("start testing")
            json_test_path = None
            for json_path in [json_test_path]:
                num_test_experiments = 1
                lambda_value = {
                    "experiment_" + str(i + 1): [] for i in range(num_test_experiments)
                }
                cam_error = {
                    "experiment_" + str(i + 1): [] for i in range(num_test_experiments)
                }
                total_error = {
                    "experiment_" + str(i + 1): [] for i in range(num_test_experiments)
                }
                all_rewards = {
                    "experiment_" + str(i + 1): [] for i in range(num_test_experiments)
                }

                for i in range(num_test_experiments):

                    print("testing experiment number: ", i)
                    lowest_iter_number = np.inf
                    replay_buffer = ReplayBuffer(
                        replay_buffer_size, random_chosing=random_sampling_buffer
                    )
                    if random_sampling_buffer is True:
                        batch_replay_buffer = ReplayBuffer(
                            batch_replay_buffer_size, random_chosing=False
                        )

                    # for general comparison
                    start_time = time.time()
                    env.reset(iter=i + num_training_experiments, json_path=json_path)
                    end_time = time.time()
                    print("env setup time:", (end_time - start_time))
                    print("classic solving")
                    classic_dict = env.classic_BA_LM(action=0.1)
                    classic_iters.append(classic_dict["iter_num"])
                    convergence_reason_classic.append(classic_dict["convergence_reason"])


                    # initializing done
                    done = False
                    env.enable_printing(True)
                    frame_idx = 0
                    batches_counter = 0
                    print("network step")
                    # Testing Loop
                    while frame_idx < max_frames:
                        state = env.reset(
                            iter=i + num_test_experiments, json_path=json_path
                        )
                        episode_reward = 0
                        if done is True:
                            frame_idx = 0
                            break
                        for step in range(max_steps):
                            if done is True:
                                break
                            start_time = time.time()
                            action = policy_net.get_action(state).detach()
                            end_time = time.time()
                            if action < 0:
                                print("Action out of range")


                            next_state, reward, done, info_dict = env.step(action.numpy())

                            print("net time:", end_time - start_time)

                            if next_state is None:
                                print("None next_state")
                            elif reward is None:
                                print("None reward")

                            if str(info_dict["lambda"]).__contains__("nan"):
                                print("bad lambda detected")
                                break
                            try:
                                lambda_value["experiment_" + str(i + 1)].append(
                                    info_dict["lambda"]
                                )
                                cam_error["experiment_" + str(i + 1)].append(
                                    info_dict["cam_error"]
                                )
                                total_error["experiment_" + str(i + 1)].append(
                                    info_dict["total_error"]
                                )
                                all_rewards["experiment_" + str(i + 1)].append(reward)
                            except:
                                continue

                            if random_sampling_buffer is False:
                                # Insert all states as a continues flow for the network to learn from
                                replay_buffer.push(state, action, reward, next_state, done)
                            else:
                                # Insert small batches that would be randomly chosen
                                if frame_idx % batch_replay_buffer_size == 0:
                                    # b_ as in batch
                                    if frame_idx == 0:
                                        # first iteration
                                        batch_replay_buffer.push(
                                            state, action, reward, next_state, done
                                        )
                                    else:
                                        (
                                            b_state,
                                            b_action,
                                            b_reward,
                                            b_next_state,
                                            b_done,
                                        ) = batch_replay_buffer.sample(
                                            batch_replay_buffer_size
                                        )
                                        replay_buffer.push(
                                            b_state,
                                            b_action,
                                            b_reward,
                                            b_next_state,
                                            b_done,
                                        )
                                        batch_replay_buffer = ReplayBuffer(
                                            batch_replay_buffer_size, random_chosing=False
                                        )
                                        # Add the last iteration ti the new batch_replay_buffer
                                        batch_replay_buffer.push(
                                            state, action, reward, next_state, done
                                        )
                                else:
                                    batch_replay_buffer.push(
                                        state, action, reward, next_state, done
                                    )

                            if frame_idx == 1000:
                                print("finished 1000 iters")

                            state = next_state
                            episode_reward += reward
                            frame_idx += 1

                            if len(replay_buffer) % max_steps > batch_size:
                                update(batch_size)

                            last_k_iters_avg += episode_reward
                            if frame_idx % 1000 == 0:
                                print(
                                    "frame: ",
                                    frame_idx,
                                    " avg reward of last 1000 iters: ",
                                    last_k_iters_avg / 1000,
                                )
                                last_k_iters_avg = 0

                            if done is True:
                                it = info_dict["iter_num"]
                                step_iters.append(it)
                                convergence_reason_step.append(
                                    info_dict["convergence_reason"]
                                )
                                break

                        rewards.append(episode_reward)

                #Print all results
                classic_iter_sum = 0
                network_iter_sum = 0
                classic_sucess_rate = 0
                network_sucess_rate = 0
                print("Classic iters", classic_iters)
                print("network iters", step_iters)
                print("Classic convergence reason", convergence_reason_classic)
                print("network convergence reason", convergence_reason_step)
                f.write("Classic, Network, Network convergence reason \n")

                rewards_avg = {
                    "experiment_"
                    + str(i + 1)
                    + "_avg_reward": sum(all_rewards["experiment_" + str(i + 1)])
                    / step_iters[i]
                    for i in range(num_test_experiments)
                }

                for w, n, w_r, n_r in zip(
                    classic_iters,
                    step_iters,
                    convergence_reason_classic,
                    convergence_reason_step,
                ):
                    classic_iter_sum += w
                    network_iter_sum += n
                    if w_r == "converged":
                        classic_sucess_rate += 1
                    if n_r == "converged":
                        network_sucess_rate += 1
                    str_to_write = (
                        str(w) + " " + str(w_r) + " " + str(n) + " " + str(n_r) + "\n"
                    )
                    f.write(str_to_write)

                avg_classic = (
                    "average classic iterations were: "
                    + str(classic_iter_sum / len(classic_iters))
                    + "\n"
                )
                f.write(avg_classic)
                avg_network = (
                    "average network iterations were: "
                    + str(network_iter_sum / len(while_iters))
                    + "\n"
                )
                f.write(avg_network)
                #print(avg_network)
                sr_classic = (
                    "Classic sucess rate: " + str(classic_sucess_rate / len(classic_iters)) + "\n"
                )
                f.write(sr_classic)
                sr_network = (
                    "Network sucess rate: "
                    + str(network_sucess_rate / len(while_iters))
                    + "\n"
                )
                f.write(sr_network)

                for i in range(num_test_experiments):
                    f.write("experiment: " + str(i + 1) + "\n")
                    f.write("Avg reward" + "\n")
                    f.write(
                        str(rewards_avg["experiment_" + str(i + 1) + "_avg_reward"]) + "\n"
                    )
                    print(
                        "avg reward:",
                        str(rewards_avg["experiment_" + str(i + 1) + "_avg_reward"]),
                    )
                    f.write("All rewards" + "\n")
                    f.write(str(all_rewards["experiment_" + str(i + 1)]) + "\n")
                    print("all rewards:" + "\n", all_rewards["experiment_" + str(i + 1)])
                    f.write("Labmda value" + "\n")
                    f.write(str(lambda_value["experiment_" + str(i + 1)]) + "\n")
                    print("Lambda values:" + "\n", lambda_value["experiment_" + str(i + 1)])
                    f.write("Cameras error" + "\n")
                    f.write(str(cam_error["experiment_" + str(i + 1)]) + "\n")
                    print("Camera errors:" + "\n", cam_error["experiment_" + str(i + 1)])
                    f.write("Total error" + "\n")
                    f.write(str(total_error["experiment_" + str(i + 1)]) + "\n")
                    print("Total errors:" + "\n", total_error["experiment_" + str(i + 1)])
        # end of "with open"
    


if __name__ == "__main__":
    main()