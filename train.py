import torch
import os
import numpy as np
import random
from trafficjam import TrafficLightAgent
import xml.etree.ElementTree as ET
from sumo_agent import SumoAgent
from sumolib import checkBinary

import torch.optim as optim

class ParaSet:
        
    RUN_COUNTS = 72000
    RUN_COUNTS_PRETRAIN = 10000
    BASE_RATIO = [10, 10]
    TRAFFIC_FILE = ["cross.2phases_rou1_switch_rou0.xml"]
    TRAFFIC_FILE_PRETRAIN = ["cross.2phases_rou1_switch_rou0.xml"]
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

RUN_COUNT = 216000
GAMMA = 0.8 

setting_memo = "one_run"

sumoBinary_nogui = checkBinary('sumo')
sumoCmd_nogui = [sumoBinary_nogui,
                 '-c',
                 r'{0}/data/{1}/cross.sumocfg'.format(
                     os.path.split(os.path.realpath(__file__))[0], setting_memo)]

sumo_agent = SumoAgent(sumoCmd_nogui)
current_time = sumo_agent.get_current_time()

memory = build_memory(4)

model = TrafficLightAgent(4)
optimizer = optim.RMSprop(model.parameters())

while current_time < RUN_COUNT:
    car_number, phase_id = sumo_agent.get_state()
    car_number = torch.from_numpy(car_number[0])
    car_number = car_number.type(dtype=torch.FloatTensor)
    phase_id = torch.Tensor([phase_id])

    # Epsilon Greedy Algorithm to choose either model's action or random action 추가하기

    action, q_values = model(car_number, phase_id)
    print(action)
    reward, action = sumo_agent.take_action(action)
    next_state = sumo_agent.get_state()

    print("action %d \t reward %f \t q_values %s" % (int(action), reward, repr(q_values)))

    memory[phase_id][action].append(torch.cat(car_number, phase_id), action, reward, next_state)

    current_time = sumo_agent.get_current_time()

    # update model
    # calculate average reward
    average_reward = np.zeros((4, 4))
    len_memory = len(memory[phase_i][action_i])
    for phase_i in range(4):
        for action_i in range(4):
            if len_memory > 0:
                list_reward = []
                for i in range(len_memory):
                    _state, _action, _reward, _ = memory[phase_i][action_i][i]
                    list_reward.append(reward)
                average_reward[phase_i][action_i]=np.average(list_reward)

    Y = []
    # get sample
    for phase_i in range(4):
        for action_i in range(4):
            sample_size = min(300, len_memory)
            sampled_memory = random.sample(memory, sample_size)
            len_sampled_memory = len(sampled_memory)

            for i in range(len_sampled_memory):
                state, action, reward, next_state = sampled_memory[i]

                if state is terminal:
                    next_estimated_reward = 0
                else:
                    next_estimated_reward = get_next_estimated_reward(next_state)
                total_reward = reward + GAMMA * next_estimated_reward
                target = np.copy(np.array([average_reward[phase_id]]))
                target[0][action] = total_reward
                Y.append(target[0])

    Y = np.array(Y)

    # Train Network



