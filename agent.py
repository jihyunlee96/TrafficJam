# -*- coding: utf-8 -*-

'''
@author: hzw77, gjz5038

Controling agent, mainly choosing actions

'''

import json
import os
import shutil


class State(object):
    # ==========================

    D_NUM_OF_VEHICLES = (12,)
    D_CUR_PHASE = (1,)
    D_NEXT_PHASE = (1,)
    D_TIME_THIS_PHASE = (1,)
    D_IF_TERMINAL = (1,)
    D_HISTORICAL_TRAFFIC = (6,)

    # ==========================

    def __init__(self, num_of_vehicles,
                 cur_phase,
                 next_phase,
                 time_this_phase,
                 if_terminal):
        self.num_of_vehicles = num_of_vehicles
        self.cur_phase = cur_phase
        self.next_phase = next_phase
        self.time_this_phase = time_this_phase

        self.if_terminal = if_terminal

        self.historical_traffic = None


class Agent(object):

    class ParaSet:
        LEARNING_RATE = 0.001 
        UPDATE_PERIOD = 300
        SAMPLE_SIZE = 300
        SAMPLE_SIZE_PRETRAIN = 3000
        BATCH_SIZE = 20 
        EPOCHS = 50 
        EPOCHS_PRETRAIN = 500
        SEPARATE_MEMORY = True
        PRIORITY_SAMPLING = False
        UPDATE_Q_BAR_FREQ = 5 
        GAMMA = 0.8 
        GAMMA_PRETRAIN = 0
        MAX_MEMORY_LEN = 1000 
        EPSILON = 0.00
        PATIENCE = 10
        PHASE_SELECTOR = True
        DDQN = False
        D_DENSE = 20
        LIST_STATE_FEATURE = ["num_of_vehicles", "cur_phase", "next_phase"]

    def __init__(self, num_phases,
                 path_set):

        self.path_set = path_set
        self.para_set = self.ParaSet()
        self.num_phases = num_phases
        self.state = None
        self.action = None
        self.memory = []
        self.average_reward = None

    def get_state(self, state, count):

        ''' set state for agent '''
        self.state = state
        return state

    def get_next_state(self, state, count):

        return state

    def choose(self, count, if_pretrain):

        ''' choose the best action for current state '''

        pass

    def remember(self, state, action, reward, next_state):
        ''' log the history separately '''

        pass

    def reset_update_count(self):

        pass

    def update_network(self, if_pretrain, use_average, current_time):
        pass

    def update_network_bar(self):
        pass

    def forget(self):
        pass

    def batch_predict(self,file_name="temp"):
        pass