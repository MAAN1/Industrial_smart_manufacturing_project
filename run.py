
import numpy as np
import math
import random
from numpy import savez_compressed
import csv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
from operator import add

import matplotlib.pyplot as plt
import pylab
import seaborn as sns
import pandas as pd

# customized
from Deep_Learning_Part import DQN
from Render_MultiAgentWS_Mask import Render_MultiAgent
from Render_Res_MultiAgentWS_Mask import Render_Res_MultiAgent
from Environment import Assembly_line_Env

DEVICE = torch.device("cpu")
env = Assembly_line_Env()
EPISODES = 100
Max_Steps = 300
Target_replace_feq = 50
class DQNAgent():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.memory_size = 30000
        self.epsilon = 1.0
        self.batch_size = 64
        self.train_start = 3000
        self.memory = deque(maxlen=self.memory_size)
        self.loss_fun= torch.nn.MSELoss()
        self.learning_step =3000
        # Online network
        self.model = DQN(state_size, action_size)
        self.model.apply(self.weights_init)
        # target network
        self.target_model = DQN(state_size, action_size)
        self.target_model.apply(self.weights_init)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform_(m.weight)

    # Action for resource assignment
    def get_action(self, state,  epsilon):
        if np.random.rand() < epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.from_numpy(state)
            state = Variable(state).float().cpu()
            q_value = self.model(state)
            _, action = torch.max(q_value, 1)
            return int(action)

        # function to track task masked state

    def get_action_mask(self, state, TasksState_mask, null_act, epsilon):
        rand_action = 0
        mask = np.concatenate([null_act, TasksState_mask])
        mask_free_position = [i for i, x in enumerate(mask) if x == 0]
        state = np.reshape(state, [1, state_size])
        state = torch.from_numpy(state)
        state = Variable(state).float().cpu()
        q_vale = self.model(state) # model call the forward function to get the prediction DL part
        pred_array = q_vale.detach().cpu().numpy()
        calculated_q_value = [] # list
        for i in range(len(mask_free_position)):
            calculated_q_value.append(pred_array[0][mask_free_position[i]])
        if np.random.rand() < epsilon:
            action = mask_free_position[random.randint(0,len(mask_free_position)-1)]
            rand_action = 1
        else:
            max_index = np.argmax(calculated_q_value)
            action = mask_free_position[max_index]

        return action, rand_action

    # Constraints checking and function calling to

    def get_action_mask_feasible(self, state, env, Tasks_State_mask, work_state, null_act, epsilon):
        list_of_masked_task = []
        num_unfeasible_taken_action = 0
        mask_temp = Tasks_State_mask
        action_task_WS = 0
        if work_state == 1:
            action_task_WS = 0
        else:
            action_task_WS, random_action = self.get_action_mask(state, mask_temp, null_act, epsilon)
            unfeasible_flag = 1
            while(unfeasible_flag == 1):
                unfeasible_flag = env.check_constraint(action_task_WS)
                if unfeasible_flag ==1:
                    mask_temp[action_task_WS-1]= 1
                    list_of_masked_task.append(action_task_WS-1)
                    num_unfeasible_taken_action += 1
        return action_task_WS, list_of_masked_task

    def update_target_model(self):
        """
        for updating target network after 300 count
        """
        self.target_model.load_state_dict(self.model.state_dict())


    def append_sample(self, state, action, reward, next_state, done):
        """
        for ReplyMemory buffer
        memory a
        """
        self.memory.append((state, action, reward, next_state, done))

    """
    train_model function calculate the current q_value, next q_next, loss calculation and 
    """

    def train_model(self):
        if self.learning_step % Target_replace_feq==0:
            self.update_target_model()
        self.learning_step +=1
        mini_batch = random.sample(self.memory, self.batch_size)
        mini_batch = np.array(mini_batch).transpose()
        states = np.vstack(mini_batch[0])
        actions = list(mini_batch[1])
        rewards = list(mini_batch[2])
        next_states = np.vstack(mini_batch[3])
        dones = mini_batch[4]
        dones = dones.astype(int)
        states = torch.Tensor(states)
        states = Variable(states).float()

        q_eval = self.model(states) # q_value

        a = torch.LongTensor(actions).view(-1, 1)
        one_hot_action = torch.FloatTensor(self.batch_size, self.action_size).zero_()
        one_hot_action.scatter_(1, a, 1)
        q_eval = torch.sum(q_eval.mul(Variable(one_hot_action)), dim=1) #

        next_states = torch.Tensor(next_states)
        next_states = Variable(next_states).float()

        q_next = self.target_model(next_states).data # q values next
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        q_target = rewards + (1 - dones) * self.discount_factor * q_next.max(1)[0]

        q_target = Variable(q_target) # next q values

        #loss = self.loss_fun(q_eval, q_target)
        loss = F.mse_loss(q_eval,q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

if __name__ == "__main__":
        learn_flag =True
        step_counter = 1
        env = Assembly_line_Env()
        state_size = env.observation_space.shape[0]
        print("state size :", state_size)

        action_size_task = env.ActionSpace_task_number.n
        print("action_size:", action_size_task)
        action_size_resource = env.ActionSpace_resource_number.n
        model_Task_WS1 = DQN(state_size, action_size_task)
        agent_Task_WS1 = DQNAgent(state_size, action_size_task)
        model_Task_WS2 = DQN(state_size, action_size_task)
        agent_Task_WS2 = DQNAgent(state_size, action_size_task)
        model_Resource_WS1 = DQN(state_size, action_size_resource)
        agent_Resource_WS1 = DQNAgent(state_size, action_size_resource)
        model_Resource_WS2 = DQN(state_size, action_size_resource)
        agent_Resource_WS2 = DQNAgent(state_size, action_size_resource)

    # writer = SummaryWriter() section

        scores, episodes, scores_1, episodes_1, results, loss_agent_1, store, job_end= [], [], [], [], [], [], [], []
        start_superlist, duration_superlist, machine_superlist,job_id_superlist = [],[],[],[]
        start_superlist_R, duration_superlist_R, machine_superlist_R,job_id_superlist_R = [],[],[],[]
        step_list, epsilon_list = [], []

        for e in range(EPISODES):
            step = 0
            done = False
            score = 0
            loss_1 = 0
            grade_to_num=0
            best_reward=0
            episode_number = []
            average_reward = 0
            random_action_counter=0
            average_reward_number=[]
            alpha = 0.13
            epsilon=math.exp(-e/(alpha*EPISODES))
            #epsilon=1.0
            state = env.reset()
            env.reset_variable()
            state = np.reshape(state, [1, state_size])
            # second loop of DQN agent training
            while not done:
                iteration = 0
                null_act = [0]
                tasks_state_mask = env.state[env.W_number:env.W_number+env.T_number]
                work_state = env.state[:env.W_number]

                action_resource_WS1 = agent_Resource_WS1.get_action(state,epsilon)
                action_resource_WS2 = agent_Resource_WS2.get_action(state,epsilon)
                action_task_WS1, list_masked_task = agent_Task_WS1.get_action_mask_feasible(state, env, tasks_state_mask, work_state[0], null_act, epsilon)
                action_task_WS2, list_masked_task = agent_Task_WS2.get_action_mask_feasible(state,env,tasks_state_mask,work_state[1],null_act,epsilon)

                check_twin = env.twin_check(action_task_WS1, action_task_WS2)
                if check_twin==1:
                    action_temp = action_task_WS1
                    tasks_state_mask[action_task_WS1-1] = 1
                    action_task_WS2, random_action = agent_Task_WS2.get_action_mask(state,tasks_state_mask,null_act,epsilon)
                    tasks_state_mask[action_temp-1] = 0
                    if random_action == 1:
                        random_action_counter +=1
                #print("Action Proposed by agnts are:", action_task_WS1, action_task_WS2, "episode:", e)
                #print("Action proposed by agents are:", action_resource_WS1, action_task_WS2, "episode:", e)
                next_state, reward, done, info = env.step(action_task_WS1, action_resource_WS1, action_task_WS2, action_resource_WS2)
                next_state = np.reshape(next_state, [1, state_size])
                #print("Reward trend during current episode:", e, "Reward", reward )

            # Memory storage
                agent_Task_WS1.append_sample(state, action_task_WS1, reward, next_state, done)
                agent_Resource_WS1.append_sample(state, action_resource_WS1, reward, next_state, done)
                agent_Task_WS2.append_sample(state, action_task_WS2, reward, next_state, done)
                agent_Resource_WS2.append_sample(state, action_resource_WS2, reward, next_state, done)
                # print("final reward for each agent:", score)

                score += reward
                #print("Final reward", score)
                state = next_state
                step +=1

                if len(agent_Task_WS1.memory) > agent_Task_WS1.train_start:

                    if learn_flag:
                        print("learning start...")
                        learn_flag = False
                    loss_1 = agent_Task_WS1.train_model()
                    #grade_to_num = torch.Tensor.detach(loss_1).numpy()
                    loss_2 = agent_Resource_WS1.train_model()
                    loss_3 = agent_Task_WS2.train_model()
                    loss_4 = agent_Resource_WS2.train_model()

                """
                score += reward
                scores_1.append(reward)
                state = next_state
                step +=1
                """

                if step == Max_Steps:

                    result = [reward, e, score]
                    scores.append(score)
                    episodes.append(e)
                    #scores_1.append(score)
                    episodes_1.append(e)
                    results.append(reward)
                    step_list.append(step)
                    epsilon_list.append(epsilon)

                    start_superlist.append(env.n_start_time)
                    duration_superlist.append(env.n_duration)
                    machine_superlist.append(env.n_bay_start)
                    job_id_superlist.append(env.n_job_id)
                    start_superlist_R.append(env.n_start_time_R)
                    duration_superlist_R.append(env.n_duration_R)
                    machine_superlist_R.append(env.n_bay_start_R)
                    job_id_superlist_R.append(env.n_job_id_R)
                    episode_number.append(e)
                    break
                if done:
                    start_superlist.append(env.n_start_time)
                    duration_superlist.append(env.n_duration)
                    machine_superlist.append(env.n_bay_start)
                    job_id_superlist.append(env.n_job_id)

                    start_superlist_R.append(env.n_start_time_R)
                    duration_superlist_R.append(env.n_duration_R)
                    machine_superlist_R.append(env.n_bay_start_R)
                    job_id_superlist_R.append(env.n_job_id_R)

                    result = [score,e,step]
                    scores.append(score)
                    episodes.append(e)
                    results.append(reward)
                    #scores_1.append(score)
                    episodes_1.append(e)
                    step_list.append(step)
                    epsilon_list.append(epsilon)

                    #job_end = env.n_start_time[env.n_job_id] + env.n_duration[env.n_job_id]

            #print("result of each episode:", result, type(result))
            print("result of each episode:", e, score)

        max_index_col = np.argmax(scores)

        res_list = []
        #for i in range(0, len(test_list1)):
        #    res_list.append(test_list1[i] + test_list2[i])
        #list_one= job_id_superlist[max_index_col]
        print(max)
        list_one=[]
        list_two=[]
        res_list =[]

        list_one = start_superlist[max_index_col]

        list_two =duration_superlist[max_index_col]

        #for i in range(0, max):
        #    res_list.append(start_superlist[i]+ duration_superlist[i])
        res_list = list(map(add, list_one, list_two))
        print("reults of two list",res_list)
        print("job_end",job_end)
        print("Job_id", job_id_superlist[max_index_col], type(job_id_superlist[max_index_col]))
        print("job start", start_superlist[max_index_col])
        print("Job duration", duration_superlist[max_index_col])
        print("Job end", job_end)
        print( "At which machine ", machine_superlist[max_index_col])


       
        # "Duration": duration_superlist[max_index_col],
        # d = {'Car': ['BMW', 'Lexus', 'Audi', 'Mercedes', 'Jaguar', 'Bentley'], 'Date_of_purchase': ['2020-10-10', '2020-10-12', '2020-10-17', '2020-10-16', '2020-10-19', '2020-10-22']
        scenerio = {"Job": job_id_superlist[max_index_col],"start":start_superlist[max_index_col],"end": res_list  }
        print("Scenerio:", scenerio)
        dataFrame = pd.DataFrame(scenerio)
        print("DataFrame...\n",dataFrame)
        dataFrame.to_csv("C:\\Users\\imran\\PycharmProjects\\Industrial_smart_manufacturing_project\\smart_industry.csv")










#-----------------#
#print(scores_1)
#print("All values:", scores, type(scores))
l_a = np.array(scores)
#print("All values to array ",l_a, type(l_a))

#tuple(map(tuple, l_a))
a_t = tuple(l_a)
#print("array to tuple", a_t)

df=[]
df = pd.DataFrame([a_t])
sns.set()
sns.lineplot(data=df.T)
#pd.DataFrame([a_t].melt(var_name='episode',value_name='reward'))
print(df.T)
#plt.show()
#for i in range(len(l_a)):
#    df.append(pd.DataFrame(l_a[i]).melt(var_name='episode',value_name='reward'))
#prin(df)
"""

#data = reward_at_each_iteration()
#print("store",store)
#print("Concatinated vales of scores:",rewards, "scores", scores,  type(rewards), type(scores))
sns.set()
#plt.plot("Secors / rewards ",EPISODES, np.array(scores))
rewards=np.vstack((scores_1)) #  Merge array
#print("after concatination:", rewards, type(rewards))
df = pd.DataFrame(rewards).melt(var_name='episode',value_name='reward') #  This conversion method is recommended
print(df)
sns.lineplot(x="episode", y="reward", data=df)
sns.lineplot(x=EPISODES,y=rewards)
#plt.plot(rewards)
plt.show()
"""

