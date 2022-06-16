import torch
import random
import numpy as np
from collections import deque
import torch.optim as optim
from torch.autograd import Variable
from P_2_RL import DQN
import torch.nn.functional as F
from environment_copy import Assembly_line_Env

env = Assembly_line_Env()
state_size = env.observation_space.shape[0]
action_size_task = env.ActionSpace_task_number.n
action_size_resource = env.ActionSpace_resource_number.n

"""
Reinforcement Learning Part (RL)
Main agent class 
working of DQN agent and why we need target network: 
our target class is dynamic, like in simple machine learning we have static class we need to learn but 
In RL observation changes,  after each action, we need target model to save Theta values  and we replace during next itteration 
or we can decide according to our need.
"""

class DQNAgent():
    def __init__(self, state_size, action_size):

        """
        Agnet parameter
        deque: is double ended data structure for storing and removing the data here we use to create the memory of agent
        :param state_size:
        :param action_size:
        below arguments need to be check
        self.load_model = False
        self.epsilon_min = 0.01
        self.explore_step = 80000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        memory_size = 2000
        batch_size = 256
        """

        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = 0.9
        self.learning_rate = 0.001
        self.memory_size = 2000
        self.epsilon = 1.0
        self.batch_size = 64
        self.train_start = 1000

        """
        RL_Agent memory creation by deque data structure 
        NN model for agent
        weight function for NN to find the optimal values to reduce the error
        target model to check the difference between prediction and actual target 
        
        Questions: What is input of NN
                   which policy agent following 
                   what is  action 
        """

        self.memory = deque(maxlen=self.memory_size)
        self.model = DQN(state_size, action_size)
        self.model.apply(self.weights_init)

        """
        target model
        """

        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

        """
        FUnctions: 
        1) weight initialization
        2) masked actions
        3) action_selection for resources 
        4) target_model_update for stablizing the training 
        5) memory creation function for agent 
        """
    #-----------------------------------------Function------------------------------------------------------------#

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform_(m.weight)

    def get_action(self, state,  epsilon_custom):

        if np.random.rand() <= epsilon_custom:
            return random.randrange(self.action_size)
        else:
            state = torch.from_numpy(state)
            state = Variable(state).float().cpu()
            q_value = self.model(state)
            _, action = torch.max(q_value, 1)
            return int(action)
    #------------------------------------------------------------------------------------------------------#

    """      
        In below function we are performing DL part calculating q_values:
        by these q values using policy we proposing available actions
        then checking pre constraint before executing in environment.
        name of function is not okay we need to change
        mask_free position: mean we have following avalable task for execution
    """

    def get_action_mask(self, state, TasksState_mask, null_act, epsilon_custom):
        rand_action = 0
        mask = np.concatenate([null_act, TasksState_mask])
        mask_free_position = [i for i, x in enumerate(mask) if x == 0]
        state = np.reshape(state, [1, state_size])
        state = torch.from_numpy(state)
        state = Variable(state).float().cpu()
        pred = self.model(state) # model call the forward function to get the prediction DL part
        pred_array = pred.detach().cpu().numpy()
        calculated_q_value = [] # list
        for i in range(len(mask_free_position)):
            calculated_q_value.append(pred_array[0][mask_free_position[i]])
        if np.random.rand() <= epsilon_custom:
            action = mask_free_position[random.randint(0,len(mask_free_position)-1)]
            rand_action = 1
        else:
            max_index = np.argmax(calculated_q_value)
            action = mask_free_position[max_index]

        """
        if rand_action==1:
            print("random action is selected by get_action_masked:", action)
        else:
            print("maxi_q values action selected by get_Action_masked:", action)
        """

        return action , rand_action

    #------------------------------------------------------------------------------------------#

    """
    In this function we check the pre constraint and wrorkstation status
    # constraint checking
    # print("first we select the action by get_action_masked:", action_task_WS) # the number of action
    # print("Second we check the pred constraint by calling the constraint function:" )
    #print(' action proposed by agent is: ',action_task_WS,'which is  unfeasible and  masked to the next iteration                                                             CHANGEEEE')
    #print("the mask is ",mask_temp,' task_state :',Tasks_State_mask, "list of masked_task state:", list_masked_task)
    #print("total number pf unfeasible actions are:", num_unfeasible_taken_action)
    """

    # -----------------------------------------------------------------------------------------#

    def get_action_mask_feasible(self, state, env, Tasks_State_mask, work_state, null_act, epsilon_custom):
        list_masked_task = []
        num_unfeasible_taken_action = 0
        mask_temp = Tasks_State_mask
        action_task_WS = 0
        if work_state ==1:
            action_task_WS = 0
        else:
            action_task_WS, random_action = self.get_action_mask(state, mask_temp, null_act, epsilon_custom)

            unfeasible_flag = 1
            while(unfeasible_flag == 1):
                action_task_WS, random_action = self.get_action_mask(state, mask_temp, null_act, epsilon_custom)
                unfeasible_flag = env.check_constraint(action_task_WS)
                if unfeasible_flag ==1:
                    mask_temp[action_task_WS-1]= 1
                    list_masked_task.append(action_task_WS-1) # indx of action
                    num_unfeasible_taken_action += 1
        return action_task_WS, list_masked_task

    # ---------------------------------------------------------------------------------------------#
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # -----------------------------------------------------------------------------------------#
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    """
    output layer of NN action selection by following some policy 
    if value of epslon is greater or = then the random number chose that action  (explore )
    otherwise choose greedy action (exploit)
    """

    """
    After memory creation agent take data from his memory to train and perform action
    """

    def train_model(self):
        mini_batch = random.sample(self.memory, self.batch_size)
        mini_batch = np.array(mini_batch).transpose()

        """
        step function (output)
        """

        states = np.vstack(mini_batch[0])
        actions = list(mini_batch[1])
        rewards = list(mini_batch[2])
        next_states = np.vstack(mini_batch[3])
        dones = mini_batch[4]
        dones = dones.astype(int)

        """
        variable prepration: 
        for storage of state, prediction and action variable.
        current q values 
        """

        states = torch.Tensor(states)
        states = Variable(states).float()
        pred = self.model(states)
        a = torch.LongTensor(actions).view(-1, 1)
        one_hot_action = torch.FloatTensor(self.batch_size, self.action_size).zero_()
        one_hot_action.scatter_(1, a, 1)
        pred = torch.sum(pred.mul(Variable(one_hot_action)), dim=1)

        """
        Next state variable for storage 
        target q values 
        loss function application
        optimizer application 
      
       
        Learning part start here
        """

        next_states = torch.Tensor(next_states)
        next_states = Variable(next_states).float()
        next_pred = self.target_model(next_states).data
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        target = rewards + (1 - dones) * self.discount_factor * next_pred.max(1)[0]
        target = Variable(target)
        self.optimizer.zero_grad()
        loss = F.mse_loss(pred,target)
        loss.backward()
        self.optimizer.step()
