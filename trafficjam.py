import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from PIL import Image

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
        self.q_values_hidden_1 = nn.Linear(20, number_of_actions) #  activation : Linear (?)
        self.linear_act_hidden_1 = nn.linear(number_of_actions, number_of_actions)

        self.seperated_hidden_2 = nn.Linear(20, 20) # activation : Sigmoid
        self.q_values_hidden_2 = nn.Linear(20, number_of_actions) #  activation : Linear (?)
        self.linear_act_hidden_2 = nn.linear(number_of_actions, number_of_actions)

        self.seperated_hidden_3 = nn.Linear(20, 20) # activation : Sigmoid
        self.q_values_hidden_3 = nn.Linear(20, number_of_actions) #  activation : Linear (?)
        self.linear_act_hidden_3 = nn.linear(number_of_actions, number_of_actions)

        self.seperated_hidden_4 = nn.Linear(20, 20) # activation : Sigmoid
        self.q_values_hidden_4 = nn.Linear(20, number_of_actions) #  activation : Linear (?)
        self.linear_act_hidden_4 = nn.linear(number_of_actions, number_of_actions)

        self.sigmoid = nn.Sigmoid()
        
        self.selector_1 = Selector()
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
        
        

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)


# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

class Selector(nn.Module):
    def __init__(self, select, **kwargs):
        super(Selector, self).__init__(**kwargs)
        self.select = select

    def forward(self, x):
        x = torch.eq(x, self.select).type(torch.FloatTensor)
        return x

