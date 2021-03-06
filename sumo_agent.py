# -*- coding: utf-8 -*-

'''
@author: hzw77, gjz5038

Interacting with traffic_light_dqn.py and map_computor.py

1) retriving values from sumo_computor.py

2) update state

3) controling logic

'''

from sys import platform
import sys
import os
import map_computor
import numpy as np
import shutil
import json
from agent import State


class SumoAgent:
    
    class ParaSet:
        MIN_PHASE_TIME = [0, 0, 0, 0]
        MIN_ACTION_TIME = 10
        REWARDS_INFO_DICT = {
                'queue_length' : [True, -0.25],
                'delay' : [True, -0.25],
                "wait_time": [True, -0.25]
        }        
    
    def __init__(self, sumo_cmd_str):

        self.para_set = self.ParaSet()

        map_computor.start_sumo(sumo_cmd_str)

        self.dic_vehicles = {}
        self.state = None
        self.current_phase = 0
        self.current_phase_duration = 0
        self.update_state()

        self.f_log_rewards = os.path.join('./data', 'log_rewards.txt')
        if not os.path.exists(self.f_log_rewards):
            f = open(self.f_log_rewards, 'w')
            list_reward_keys = np.sort(list(self.ParaSet.REWARDS_INFO_DICT.keys()))
            head_str = "count, action," + ",".join(list_reward_keys) + '\n'
            f.write(head_str)
            f.close()

    def end_sumo(self):
        map_computor.end_sumo()

    def get_observation(self):
        return self.state

    def get_current_time(self):
        return map_computor.get_current_time()

    def get_current_phase(self):
        return self.current_phase

    def take_action(self, action, phase):
        current_phase_number = self.get_current_phase()
        rewards_detail_dict_list = []
        if int(action) == int(phase):
            changed = False
        else:
            changed = True
        # 현재 신호가 최소 지속 시간을 넘지 않았다면 a
        # MIN_ACTION_TIME 까지 돌아라(5까지)
        for i in range(self.ParaSet.MIN_ACTION_TIME):
            # action time 동안에 
            action_in_second = self.get_current_phase()
            current_phase_number = self.get_current_phase()
            #### 이 부분 바꿔야함!###
            # a가 바꾸는 거라면 일단 첫 액션은 바꾸는 걸로 해라
            if changed and i == 0:
                action_in_second = action
            # 현재 상황에서 actioninsecond을 주었을 때의 결과
            self.current_phase, self.current_phase_duration, self.vehicle_dict = map_computor.run(action=action_in_second,
                                                                               current_phase=current_phase_number,
                                                                               current_phase_duration=self.current_phase_duration,
                                                                               vehicle_dict=self.dic_vehicles,
                                                                               rewards_info_dict=self.ParaSet.REWARDS_INFO_DICT,
                                                                               f_log_rewards=self.f_log_rewards,
                                                                               changed=changed,
                                                                               rewards_detail_dict_list=rewards_detail_dict_list)  # run 1s SUMO
            changed = False

        reward = self.cal_reward_from_list(rewards_detail_dict_list)

        self.update_state()

        return reward, action

    def update_state(self):
        status_tracker = map_computor.status_calculator()
        # state를 인풋으로 받는 위치
        self.state = State(
            # 필요한 부분
            num_of_vehicles=np.reshape(np.array(status_tracker[1]), newshape=(1, 14)),
            cur_phase=np.reshape(np.array([self.current_phase]), newshape=(1, 1)),
            next_phase=np.reshape(np.array([(self.current_phase + 1) % len(self.ParaSet.MIN_PHASE_TIME)]), newshape=(1, 1)),
            time_this_phase=np.reshape(np.array([self.current_phase_duration]), newshape=(1, 1)),
        )
    def get_state(self):
        return self.state.num_of_vehicles, self.current_phase
    
    def cal_reward(self, action):
        # get directly from sumo
        reward, reward_detail_dict = map_computor.get_rewards_from_sumo(self.dic_vehicles, action, self.para_set.REWARDS_INFO_DICT)
        return reward * 0.2, reward_detail_dict

    def cal_reward_from_list(self, reward_detail_dict_list):
        reward = map_computor.get_rewards_from_dict_list(reward_detail_dict_list)
        return reward * 0.2

    def terminate_sumo(self):
        map_computor.terminate_sumo()
        return


if __name__ == '__main__':
    pass
