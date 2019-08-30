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

class Vehicles:
    initial_speed = 5.0

    def __init__(self):
        # add what ever you need to maintain
        self.id = None
        self.speed = None
        self.wait_time = None
        self.stop_count = None
        self.enter_time = None
        self.has_read = False
        self.first_stop_time = -1
        self.entering = True


class SumoAgent:
    
    class ParaSet:
        MIN_PHASE_TIME = [0, 0]
        MIN_ACTION_TIME = 5,
        REWARDS_INFO_DICT = {
            queue_length = [true, -1],
        }        
    
    def __init__(self, sumo_cmd_str, path_set):

        self.path_set = path_set

        self.para_set = self.ParaSet()

        map_computor.start_sumo(sumo_cmd_str)

        self.dic_vehicles = {}
        self.state = None
        self.current_phase = 0
        self.current_phase_duration = 0
        self.update_state()

    def end_sumo(self):
        map_computor.end_sumo()

    def get_observation(self):
        return self.state

    def get_current_time(self):
        return map_computor.get_current_time()

    def get_current_phase(self):
        return self.current_phase

    def take_action(self, action):
        current_phase_number = self.get_current_phase()
        rewards_detail_dict_list = []
        # 현재 신호가 최소 지속 시간을 넘지 않았다면 action은 유지
        if (self.current_phase_duration < self.para_set.MIN_PHASE_TIME[current_phase_number]):
            action = 0
        # MIN_ACTION_TIME 까지 돌아라(5까지)
        for i in range(self.para_set.MIN_ACTION_TIME):
            # action time 동안에 
            action_in_second = 0
            current_phase_number = self.get_current_phase()
            # a가 바꾸는 거라면 일단 첫 액션은 바꾸는 걸로 해라
            if action == 1 and i == 0:
                action_in_second = 1
            # 현재 상황에서 actioninsecond을 주었을 때의 결과
            self.current_phase, self.current_phase_duration, self.vehicle_dict = map_computor.run(action=action_in_second,
                                                                               current_phase=current_phase_number,
                                                                               current_phase_duration=self.current_phase_duration,
                                                                               vehicle_dict=self.dic_vehicles,
                                                                               rewards_info_dict=self.para_set.REWARDS_INFO_DICT,
                                                                               rewards_detail_dict_list=rewards_detail_dict_list)  # run 1s SUMO

        #reward, reward_detail_dict = self.cal_reward(action)
        reward = self.cal_reward_from_list(rewards_detail_dict_list)
        self.update_state()

        return reward, action

    def update_state(self):
        status_tracker = map_computor.status_calculator()
        # state를 인풋으로 받는 위치
        self.state = State(
            # 필요한 부분
            num_of_vehicles=np.reshape(np.array(status_tracker[1]), newshape=(1, 12)),
            cur_phase=np.reshape(np.array([self.current_phase]), newshape=(1, 1)),
            next_phase=np.reshape(np.array([(self.current_phase + 1) % len(self.para_set.MIN_PHASE_TIME)]), newshape=(1, 1)),
            time_this_phase=np.reshape(np.array([self.current_phase_duration]), newshape=(1, 1)),
        )
    def get_state(self):
        return self.state.num_of_vehicles, self.current_phase
    
    def cal_reward(self, action):
        # get directly from sumo
        reward, reward_detail_dict = map_computor.get_rewards_from_sumo(self.dic_vehicles, action, self.para_set.REWARDS_INFO_DICT)
        return reward*(1-0.8), reward_detail_dict

    def cal_reward_from_list(self, reward_detail_dict_list):
        reward, reward_detail_dict = map_computor.get_rewards_from_sumo(self.dic_vehicles, action, self.para_set.REWARDS_INFO_DICT)
        # reward = map_computor.get_rewards_from_dict_list(reward_detail_dict_list)
        return reward


if __name__ == '__main__':
    pass