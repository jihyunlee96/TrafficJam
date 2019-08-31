import torch
import os
import numpy as np
import random
from trafficjam import TrafficLightAgent
import xml.etree.ElementTree as ET
from sumo_agent import SumoAgent
from sumolib import checkBinary

import torch.optim as optim
import torch.nn as nn

class ParaSet:
        
    RUN_COUNTS = 216000
    BASE_RATIO = [10, 10]
    TRAFFIC_FILE = ["cross.2phases_rou1_switch_rou0.xml"]
    MODEL_NAME = "TrafficJAM"

def build_memory(num_phases):
    memory = []
    for i in range(num_phases):
        memory.append([[] for j in range(num_phases)])
    return memory

def _set_traffic_file(sumo_config_file_tmp_name, sumo_config_file_output_name, list_traffic_file_name):

    # update sumocfg
    sumo_cfg = ET.parse(sumo_config_file_tmp_name)
    config_node = sumo_cfg.getroot()
    input_node = config_node.find("input")
    for route_files in input_node.findall("route-files"):
        input_node.remove(route_files)
    input_node.append(
        ET.Element("route-files", attrib={"value": ",".join(list_traffic_file_name)}))
    sumo_cfg.write(sumo_config_file_output_name)

def set_traffic_file(self):
    _set_traffic_file(
        os.path.join("./data", "cross.sumocfg"),
        os.path.join("./data", "cross.sumocfg"),
        ParaSet.TRAFFIC_FILE)
    for file_name in ParaSet.TRAFFIC_FILE:
        shutil.copy(
            os.path.join("./data", file_name),
            os.path.join("./data", file_name))

def unison_shuffled_copies(states, target, sample_weight):
    p = np.random.permutation(len(target))
    new_states = []
    new_states.append(states[p])
    return new_states, target[p], sample_weight[p]

RUN_COUNT = 216000
GAMMA = 0.8 
UPDATE_TARGET_NET_FREQ = 5

setting_memo = "one_run"

sumoBinary = checkBinary('sumo-gui')
sumoCmd = [sumoBinary,
                 '-c',
                 r'{0}/data/{1}/cross.sumocfg'.format(
                     os.path.split(os.path.realpath(__file__))[0], setting_memo)]
sumoBinary_nogui = checkBinary('sumo')
sumoCmd_nogui = [sumoBinary_nogui,
                 '-c',
                 r'{0}/data/{1}/cross.sumocfg'.format(
                     os.path.split(os.path.realpath(__file__))[0], setting_memo)]

# 이걸 바꾸면 GUI On/Off를 조절할 수 있다.
sumo_agent = SumoAgent(sumoCmd)
#sumo_agent = SumoAgent(sumoCmd_nogui)

current_time = sumo_agent.get_current_time()

num_phases = 2

memory = build_memory(num_phases)
target_net_outdated = 0

policy_net = TrafficLightAgent(num_phases)
target_net = TrafficLightAgent(num_phases)
target_net.load_state_dict(policy_net.state_dict())

def get_next_estimated_q_values(next_state, action):
    i, next_estimated_q_values = target_net(next_state, action)
    return next_estimated_q_values[i]

optimizer = optim.RMSprop(policy_net.parameters())

update_outdated = current_time
while current_time < RUN_COUNT:
    car_number, phase_id = sumo_agent.get_state()
    car_number = torch.from_numpy(car_number[0])
    car_number = car_number.type(dtype=torch.FloatTensor)
    phase_id = torch.Tensor([phase_id])

    print(car_number)

    # Epsilon Greedy Algorithm to choose either policy_net's action or random action 추가하기

    input_state = torch.cat((car_number, phase_id), 0)
    action, q_values = policy_net(input_state, phase_id)
    reward, action = sumo_agent.take_action(action, phase_id)
    next_state = sumo_agent.get_state()

    print("action %d \t reward %f \t q_values %s" % (int(action), reward, repr(q_values)))

    memory[int(phase_id)][action].append([torch.cat((car_number, phase_id), 0), action, reward, next_state])

    current_time = sumo_agent.get_current_time()



    update_outdated = current_time
    # update policy_net
    # calculate average reward
    average_reward = np.zeros((num_phases, num_phases))
    for phase_i in range(num_phases):
        for action_i in range(num_phases):
            len_memory = len(memory[phase_i][action_i])
            if len_memory > 0:
                list_reward = []
                for i in range(len_memory):
                    _state, _action, _reward, _ = memory[phase_i][action_i][i]
                    list_reward.append(_reward)
                average_reward[phase_i][action_i]=np.average(list_reward)

    sampled_states = []
    sampled_target = []
    # get sample
    for phase_i in range(num_phases):
        for action_i in range(num_phases):
            len_memory = len(memory[phase_i][action_i])
            sample_size = min(300, len_memory)
            sampled_memory = random.sample(memory[phase_i][action_i], sample_size)
            len_sampled_memory = len(sampled_memory)
            
            for i in range(len_sampled_memory):
                state, action, reward, next_state = sampled_memory[i]
                next_action = torch.Tensor([next_state[1]])
                next_state = torch.from_numpy(next_state[0][0])
                next_state = next_state.type(dtype=torch.FloatTensor)
                next_state = torch.cat((next_state, next_action), 0)

                next_estimated_q_values = get_next_estimated_q_values(next_state, next_action)
                total_reward = reward + GAMMA * next_estimated_q_values
                target = np.copy(np.array([average_reward[int(phase_id[0])]]))
                target[0][action] = total_reward.detach().numpy()
                sampled_target.append(target[0])
                sampled_states.append(state.numpy())

    sampled_states = np.array(sampled_states)
    sampled_target = np.array(sampled_target)
    sampled_weight = np.ones(len(sampled_target))
    sampled_states, sampled_target, _ = unison_shuffled_copies(sampled_states, sampled_target, sampled_weight)

    loss = nn.MSELoss()

    # Train Network
    for epoch in range(50):
        for i in range(len(sampled_states)):
            temp_state = torch.from_numpy(sampled_states[i][0])
            temp_state = temp_state.type(dtype=torch.FloatTensor)
            temp_phase = torch.Tensor([sampled_states[i][0][12]])
            action, q_values = policy_net(temp_state, temp_phase)

            temp_target = torch.from_numpy(sampled_target[i])
            temp_target = temp_target.type(dtype=torch.FloatTensor)
            output = loss(q_values, temp_target)
            optimizer.zero_grad()
            output.backward()
            for param in policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()

    target_net_outdated += 1

    if target_net_outdated >= UPDATE_TARGET_NET_FREQ:
        target_net.load_state_dict(policy_net.state_dict())
        target_net_outdated = 0


