import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import ToTensor

device = torch.cuda.set_device(0)

class TrafficLightAgent(nn.Module):
    '''
    Network Structure
    ==================

    Input: Number of Cars (each bit = number of cars in one lane), Signal Phase (Phase ID / ex. WE-Gr = 0, NS-Gr = 1)

    1. Concat the Number of Cars and Signal Phase
    2. Through Fully Connected Layer, create Embedded Input
    3. There are several rules, seperated by phases, that takes Embedded Input and outputs q_values of it
    4. For each Phase, there are seperate Selectors, which selects signal phase
    5. Multiply q_values and Selector's result
    6. Add it to list
    7. After all phases tested, add all of them
    8. With the result, through argmax, choose the action
    '''

    def __init__(self, num_actions):
        super(TrafficLightAgent, self).__init__()

        self.shared_hidden_1 = nn.Linear(15, 20) # activation : Sigmoid

        # Phase Gate - seperated routes by phases
        self.seperated_hidden_1 = nn.Linear(20, 20) # activation : Sigmoid
        self.q_values_hidden_1 = nn.Linear(20, num_actions) #  activation : Linear (?)
        self.linear_act_hidden_1 = nn.Linear(num_actions, num_actions)
        tensor1 = torch.Tensor([0])
        tensor1 = tensor1.to(device)
        self.selector_hidden_1 = Selector(tensor1)

        self.seperated_hidden_2 = nn.Linear(20, 20) # activation : Sigmoid
        self.q_values_hidden_2 = nn.Linear(20, num_actions) #  activation : Linear (?)
        self.linear_act_hidden_2 = nn.Linear(num_actions, num_actions)
        tensor2 = torch.Tensor([1])
        tensor2 = tensor2.to(device)
        self.selector_hidden_2 = Selector(tensor2)

        self.seperated_hidden_3 = nn.Linear(20, 20) # activation : Sigmoid
        self.q_values_hidden_3 = nn.Linear(20, num_actions) #  activation : Linear (?)
        self.linear_act_hidden_3 = nn.Linear(num_actions, num_actions)
        tensor3 = torch.Tensor([2])
        tensor3 = tensor3.to(device)
        self.selector_hidden_3 = Selector(tensor3)

        self.seperated_hidden_4 = nn.Linear(20, 20) # activation : Sigmoid
        self.q_values_hidden_4 = nn.Linear(20, num_actions) #  activation : Linear (?)
        self.linear_act_hidden_4 = nn.Linear(num_actions, num_actions)
        tensor4 = torch.Tensor([3])
        tensor4 = tensor4.to(device)
        self.selector_hidden_4 = Selector(tensor4)

        self.sigmoid = nn.Sigmoid()
        
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, phase):

        # Fully Connected Layer to create Embedded Input
        shared_hidden_1 = self.sigmoid(self.shared_hidden_1(x))

        # 4 different hidden layers. Each of them take care of one phase
        seperated_hidden_1 = self.sigmoid(self.seperated_hidden_1(shared_hidden_1))
        seperated_hidden_2 = self.sigmoid(self.seperated_hidden_2(shared_hidden_1))
        seperated_hidden_3 = self.sigmoid(self.seperated_hidden_3(shared_hidden_1))
        seperated_hidden_4 = self.sigmoid(self.seperated_hidden_4(shared_hidden_1))

        q_values_hidden_1 = self.linear_act_hidden_1(self.q_values_hidden_1(seperated_hidden_1))
        q_values_hidden_2 = self.linear_act_hidden_2(self.q_values_hidden_2(seperated_hidden_2))
        q_values_hidden_3 = self.linear_act_hidden_3(self.q_values_hidden_1(seperated_hidden_3))
        q_values_hidden_4 = self.linear_act_hidden_4(self.q_values_hidden_2(seperated_hidden_4))
        
        selector_hidden_1 = self.selector_hidden_1(phase)
        selector_hidden_2 = self.selector_hidden_2(phase)
        selector_hidden_3 = self.selector_hidden_3(phase)
        selector_hidden_4 = self.selector_hidden_4(phase)

        multiplied_hidden_1 = torch.mul(q_values_hidden_1, selector_hidden_1)
        multiplied_hidden_2 = torch.mul(q_values_hidden_2, selector_hidden_2)
        multiplied_hidden_3 = torch.mul(q_values_hidden_3, selector_hidden_3)
        multiplied_hidden_4 = torch.mul(q_values_hidden_4, selector_hidden_4)

        final_q_values = multiplied_hidden_1 + multiplied_hidden_2 + multiplied_hidden_3 + multiplied_hidden_4

        q_value, action = torch.max(final_q_values, 0)

        return action, final_q_values

class Selector(nn.Module):
    def __init__(self, select, **kwargs):
        super(Selector, self).__init__(**kwargs)
        self.select = select

    def forward(self, x):
        x = torch.eq(x, self.select).type(torch.FloatTensor)
        return x

