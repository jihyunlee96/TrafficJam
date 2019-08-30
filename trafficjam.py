import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def __init__(self, h, w, outputs):
        super(TrafficLightAgent, self).__init__()

        self.shared_hidden_1 = nn.Linear(18, 20) # activation : Sigmoid

        # Phase Gate - seperated routes by phases
        self.seperated_hidden_1 = nn.Linear(20, 20) # activation : Sigmoid
        self.q_values_hidden_1 = nn.Linear(20, 4) #  activation : Linear (?)
        self.linear_act_hidden_1 = nn.linear(4, 4)
        self.selector_hidden_1 = Selector(torch.Tensor([0]))

        self.seperated_hidden_2 = nn.Linear(20, 20) # activation : Sigmoid
        self.q_values_hidden_2 = nn.Linear(20, 4) #  activation : Linear (?)
        self.linear_act_hidden_2 = nn.linear(4, 4)
        self.selector_hidden_2 = Selector(torch.Tensor([1]))

        self.seperated_hidden_3 = nn.Linear(20, 20) # activation : Sigmoid
        self.q_values_hidden_3 = nn.Linear(20, 4) #  activation : Linear (?)
        self.linear_act_hidden_3 = nn.linear(4, 4)
        self.selector_hidden_3 = Selector(torch.Tensor([2]))

        self.seperated_hidden_4 = nn.Linear(20, 20) # activation : Sigmoid
        self.q_values_hidden_4 = nn.Linear(20, 4) #  activation : Linear (?)
        self.linear_act_hidden_4 = nn.linear(4, 4)
        self.selector_hidden_4 = Selector(torch.Tensor([3]))

        self.sigmoid = nn.Sigmoid()
        
        self.multiply

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, car_number, phase):
        # Concat number of cars and current phase id
        x = torch.cat((car_number, phase), 0)

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

        return final_q_values

class Selector(nn.Module):
    def __init__(self, select, **kwargs):
        super(Selector, self).__init__(**kwargs)
        self.select = select

    def forward(self, x):
        x = torch.eq(x, self.select).type(torch.FloatTensor)
        return x

