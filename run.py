import pylab
import math
import numpy as np
from Deep_Learning_Part import DQN
from Rein_Learning_Part import DQNAgent
from Render_MultiAgentWS_Mask import Render_MultiAgent
from Render_Res_MultiAgentWS_Mask import Render_Res_MultiAgent
from Environment import Assembly_line_Env
from numpy import savez_compressed
EPISODES = 50
Max_Steps = 65

if __name__ == "__main__":
    env = Assembly_line_Env()
    state_size = env.observation_space.shape[0]
    print("state size :", state_size)

    action_size_task = env.ActionSpace_task_number.n
    print("action_size:", action_size_task)
    action_size_resource = env.ActionSpace_resource_number.n

    model_Task_WS1 = DQN(state_size, action_size_task)
    agent_Task_WS1 = DQNAgent(state_size, action_size_task)

    model_Resource_WS1 = DQN(state_size, action_size_resource)
    agent_Resource_WS1 = DQNAgent(state_size, action_size_resource)

    model_Task_WS2 = DQN(state_size, action_size_task)
    agent_Task_WS2 = DQNAgent(state_size, action_size_task)

    model_Resource_WS2 = DQN(state_size, action_size_resource)
    agent_Resource_WS2 = DQNAgent(state_size, action_size_resource)

    scores, episodes, scores_1, episodes_1, results = [], [], [], [], []
    start_superlist, duration_superlist, machine_superlist,job_id_superlist = [],[],[],[]
    start_superlist_R, duration_superlist_R, machine_superlist_R,job_id_superlist_R = [],[],[],[]
    step_list, epsilon_list = [], []

    """
    DQN training agent 
    """
    for e in range(EPISODES):   # for each episode
        step = 0
        done = False
        score = 0
        alpha = 0.13
        epsilon_custom =math.exp(-e/(alpha*EPISODES))
        #epsilon_custom=0.9
        state = env.reset()
        env.reset_variable()
        state = np.reshape(state, [1, state_size])
        # second loop of DQN agent training
        while not done:
            iteration = 0
            null_act = [0]
            tasks_state_mask = env.state[env.WorkstationsNumber:env.WorkstationsNumber+env.TasksNumber]
            work_state = env.state[:env.WorkstationsNumber]
            action_resource_WS1 = agent_Resource_WS1.get_action(state,epsilon_custom)
            action_resource_WS2 = agent_Resource_WS2.get_action(state,epsilon_custom)

            action_task_WS1, list_masked_task = agent_Task_WS1.get_action_mask_feasible(state, env, tasks_state_mask, work_state[0], null_act, epsilon_custom)
            action_task_WS2, list_masked_task = agent_Task_WS2.get_action_mask_feasible(state,env,tasks_state_mask,work_state[1],null_act,epsilon_custom)

            check_twin = env.twin_check(action_task_WS1, action_task_WS2)
            if check_twin==1:
                action_temp = action_task_WS1
                tasks_state_mask[action_task_WS1-1] = 1
                action_task_WS2, random_action = agent_Task_WS2.get_action_mask(state,tasks_state_mask,null_act,epsilon_custom)
                tasks_state_mask[action_temp-1] = 0

            print("Step #:", step, " of episode", e)
            #print("Final action selected for step :\n", "action 1:" ,action_task_WS1, "\n" , "action 2:", action_task_WS2)
            next_state, reward, done, info = env.step(action_task_WS1, action_resource_WS1, action_task_WS2, action_resource_WS2)
            next_state = np.reshape(next_state, [1, state_size])

            agent_Task_WS1.append_sample(state, action_task_WS1, reward, next_state, done)
            agent_Resource_WS1.append_sample(state, action_resource_WS1, reward, next_state, done)
            agent_Task_WS2.append_sample(state, action_task_WS2, reward, next_state, done)
            agent_Resource_WS2.append_sample(state, action_resource_WS2, reward, next_state, done)

            if len(agent_Task_WS1.memory) >= agent_Task_WS1.train_start:
                agent_Task_WS1.train_model()
                agent_Resource_WS1.train_model()
                agent_Task_WS2.train_model()
                agent_Resource_WS2.train_model()

            score += reward
            state = next_state
            step +=1

            if step == Max_Steps:
                agent_Task_WS1.update_target_model()
                agent_Resource_WS1.update_target_model()
                agent_Task_WS2.update_target_model()
                agent_Resource_WS2.update_target_model()
                result = [score,e,step]
                scores.append(score)
                episodes.append(e)
                scores_1.append(score)
                episodes_1.append(e)
                results.append(result)
                step_list.append(step)
                epsilon_list.append(epsilon_custom)

                start_superlist.append(env.n_start_time)
                duration_superlist.append(env.n_duration)
                machine_superlist.append(env.n_bay_start)
                job_id_superlist.append(env.n_job_id)
                start_superlist_R.append(env.n_start_time_R)
                duration_superlist_R.append(env.n_duration_R)
                machine_superlist_R.append(env.n_bay_start_R)
                job_id_superlist_R.append(env.n_job_id_R)

                # print("Current episode:", e, "  score:", score, "  memory length:",len(agent_Task_WS1.memory), " epsilon:", epsilon_custom)
                break
            # after each episode i am updating the model
            if done:
                agent_Task_WS1.update_target_model()
                agent_Task_WS2.update_target_model()
                agent_Resource_WS1.update_target_model()
                agent_Resource_WS2.update_target_model()

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
                results.append(result)
                scores_1.append(score)
                episodes_1.append(e)
                step_list.append(step)
                epsilon_list.append(epsilon_custom)

    max_index_col = np.argmax(scores)
    print("Best Result", max_index_col, "Reward", scores[max_index_col], "Makspane", step_list[max_index_col])
    print("Initial starting point of each task start_superlist[max_index_col]", start_superlist[max_index_col])
    print(" Busy duration of workstation  duration_superlist[max_index_col]", duration_superlist[max_index_col])
    print(" Workstation Ids  machine_superlist[max_index_col]) ", machine_superlist[max_index_col])
    print("Task Ids job_id_superlist[max_index_col]", job_id_superlist[max_index_col])

    pylab.figure(1)
    pylab.plot(episodes, scores, 'b', linewidth=0.1, markersize=1)
    pylab.figure(2)
    pylab.plot(episodes, step_list, 'r', linewidth=0.1, markersize=1)
    pylab.figure(3)
    pylab.plot(episodes, epsilon_list, 'g', linewidth=1, markersize=1)

    Render_MultiAgent(start_superlist[max_index_col], duration_superlist[max_index_col], machine_superlist[max_index_col], job_id_superlist[max_index_col])
    
    Render_Res_MultiAgent(start_superlist_R[max_index_col],duration_superlist_R[max_index_col], machine_superlist_R[max_index_col], job_id_superlist_R[max_index_col])

    pylab.show()
