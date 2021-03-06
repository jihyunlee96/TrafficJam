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
from torchvision.transforms import ToTensor

device = torch.cuda.set_device(0)

class ParaSet:
        
    RUN_COUNTS = 216000
    BASE_RATIO = [10, 10]
    TRAFFIC_FILE = ["cross.rou_0.xml", "cross.rou_1.xml", "cross.rou_2.xml", "cross.rou_3.xml", "cross.rou_4.xml", "cross.rou_5.xml", "cross.rou_1_low.xml", "cross.rou_2_low.xml", "cross.rou_3_low.xml", "cross.rou_4_low.xml", "cross.rou_5_low.xml", "cross.rou_1_less.xml", "cross.rou_2_less.xml", "cross.rou_3_less.xml", "cross.rou_4_less.xml", "cross.rou_5_less.xml"]
    SUMOCFG_FILE = ["cross_0.sumocfg", "cross_1.sumocfg", "cross_2.sumocfg", "cross_3.sumocfg", "cross_4.sumocfg", "cross_5.sumocfg", "cross_1_low.sumocfg", "cross_2_low.sumocfg", "cross_3_low.sumocfg", "cross_4_low.sumocfg", "cross_5_low.sumocfg", "cross_1_less.sumocfg", "cross_2_less.sumocfg", "cross_3_less.sumocfg", "cross_4_less.sumocfg", "cross_5_less.sumocfg"]
    MODEL_NAME = "TrafficJAM"
    EPSILON = 0.05

def build_memory(num_phases):
    memory = []
    for i in range(num_phases):
        memory.append([[] for j in range(num_phases)])
    return memory

'''

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
    index = random.randint(0, 5)
    traffic_file = [ParaSet.TRAFFIC_FILE[index]]
    sumocfg_file = ParaSet.SUMOCFG_FILE[index]
    _set_traffic_file(
        os.path.join("./data/original_run", sumocfg_file),
        os.path.join("./data/original_run", sumocfg_file),
        traffic_file)
    for file_name in traffic_file:
        shutil.copy(
            os.path.join("./data/original_run", file_name),
            os.path.join("./data/original_run", file_name))
'''

def unison_shuffled_copies(states, target, sample_weight):
    p = np.random.permutation(len(target))
    new_states = []
    new_states.append(states[p])
    return new_states, target[p], sample_weight[p]

RUN_COUNT = 216000
GAMMA = 0.8 
UPDATE_TARGET_NET_FREQ = 5

setting_memo = "original_run"


# 이걸 바꾸면 GUI On/Off를 조절할 수 있다.
#sumo_agent = SumoAgent(sumoCmd)
step_count = 0

for entire_epoch in range(500):

    if entire_epoch < 250:
        index = random.randint(11, 15)
    elif entire_epoch < 400:
        index = random.randint(6, 15)
    else:
        index = random.randint(0, 15)
    traffic_file = [ParaSet.TRAFFIC_FILE[index]]
    sumocfg_file = ParaSet.SUMOCFG_FILE[index]

    sumoBinary = checkBinary('sumo-gui')
    sumoCmd = [sumoBinary,
                     '-c',
                     r'{0}/data/{1}/{2}'.format(
                         os.path.split(os.path.realpath(__file__))[0], setting_memo, sumocfg_file)]
    sumoBinary_nogui = checkBinary('sumo')
    sumoCmd_nogui = [sumoBinary_nogui,
                     '-c',
                     r'{0}/data/{1}/{2}'.format(
                         os.path.split(os.path.realpath(__file__))[0], setting_memo, sumocfg_file)]

    sumo_agent = SumoAgent(sumoCmd_nogui)
    current_time = sumo_agent.get_current_time()

    num_phases = 4

    memory = build_memory(num_phases)
    target_net_outdated = 0

    policy_net = TrafficLightAgent(num_phases).to(device)
    if entire_epoch != 0:
        policy_net.load_state_dict(torch.load("trafficjam.weight"))
    target_net = TrafficLightAgent(num_phases).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    policy_net.train()
    target_net.eval()

    def get_next_estimated_q_values(next_state, action):
        i, next_estimated_q_values = target_net(next_state, action)
        return next_estimated_q_values[i]

    optimizer = optim.RMSprop(policy_net.parameters())

    update_outdated = current_time

    while current_time < 750:
        car_number, phase_id = sumo_agent.get_state()
        car_number = torch.from_numpy(car_number[0])
        car_number = car_number.type(dtype=torch.FloatTensor)
        car_number = car_number.to(device)
        phase_id = torch.Tensor([phase_id])
        phase_id = phase_id.to(device)

        # Epsilon Greedy Algorithm to choose either policy_net's action or random action 추가하기

        input_state = torch.cat((car_number, phase_id), 0)
        action, q_values = policy_net(input_state, phase_id)

        if random.random() <= ParaSet.EPSILON:  # continue explore new Random Action
            action = random.randrange(len(q_values))
            print("##Explore")

        reward, action = sumo_agent.take_action(action, phase_id)
        next_state = sumo_agent.get_state()

        print("action %d \t reward %f \t q_values %s" % (int(action), reward, repr(q_values)))

        memory[int(phase_id)][action].append([torch.cat((car_number, phase_id), 0), action, reward, next_state])

        current_time = sumo_agent.get_current_time()

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

        if current_time - update_outdated < 300:
            continue
        else:

            loss = nn.MSELoss()

            count_stop = 0
            least_loss = -1000
            # Train Network
            for epoch in range(50):
                if count_stop >= 20:
                    break
                for i in range(len(sampled_states)):
                    temp_state = torch.from_numpy(sampled_states[i][0])
                    temp_state = temp_state.type(dtype=torch.FloatTensor)
                    temp_phase = torch.Tensor([sampled_states[i][0][12]])
                    action, q_values = policy_net(temp_state, temp_phase)

                    temp_target = torch.from_numpy(sampled_target[i])
                    temp_target = temp_target.type(dtype=torch.FloatTensor)
                    output = loss(q_values, temp_target)
                    if output > least_loss:
                        least_loss = output
                        count_stop = 0
                    else:
                        count_stop += 1

                    optimizer.zero_grad()
                    output.backward()
                    for param in policy_net.parameters():
                        if param.grad is not None:
                            param.grad.data.clamp_(-1, 1)
                    optimizer.step()

            target_net_outdated += 1

            if target_net_outdated >= UPDATE_TARGET_NET_FREQ:
                target_net.load_state_dict(policy_net.state_dict())
                target_net_outdated = 0

            if current_time > 670:
                sumo_agent.terminate_sumo()
                if entire_epoch * current_time > step_count + 25000:
                    step_count += 25000
                torch.save(policy_net.state_dict(), './trafficjam.weight')
                torch.save(policy_net.state_dict(), './trafficjam.{}.weight'.format(str(step_count)))
                print("Saved!")
                break
    print("End of epoch {}".format(str(entire_epoch)))



