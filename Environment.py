import gym
import numpy as np
from gym import spaces
from Task_Res_Action_assignment import Action_assignment, Resource_assignmnet_action
class Assembly_line_Env(gym.Env):
	def __init__(self):
		self.W_number = 2
		self.T_number = 15
		self.R_number = 3
		self.A_T_W_1 = Action_assignment(self.T_number)
		self.A_R_W_1 = Resource_assignmnet_action(self.R_number)
		self.A_T_W_2 = Action_assignment(self.T_number)
		self.A_R_W_2 = Resource_assignmnet_action(self.R_number)

		# gym part action space for task and resource
		high = np.full(self.W_number + self.T_number + self.W_number*self.R_number+1, np.finfo(np.float32).max)
		self.ActionSpace_task_number = spaces.Discrete(self.T_number+1)
		self.ActionSpace_resource_number = spaces.Discrete(self.R_number+1)

		# Observation space in our case
		self.observation_space = spaces.Box(-high, high, dtype=np.float32)
		self.state = None
		self.steps_beyond_done = None
		# Task Resource Constraint T_R_C, we need to import from excl file still pending
		self.T_R_C = [[1,0,0],[0,1,0], [0,1,0],[1,0,0], [0,0,1],[0,0,1], [0,1,0],[0,1,0] , [0,0,1], [1,0,0], [1,0,0], [0,1,0], [0,1,0], [0,0,1], [1,0,0]]
		self.Tasks_duration=[5,3,8,7,6,5,4,4,8,5,10,6,5,4,5]
		self.T_P_C = [
					[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
					[-1,0,0,0,-1,0,-1,-1,-1,0,0,0,0,0 ,0],[0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0],
					[0,0,1,0,0,0,0,-1,0,0,0,0,0,0,0],[-1,-1,-1,-1,-1,0,-1,-1,-1,0,0,0,0,0,0],
					[-1,0,1,0,-1,1,0,-1,0,0,0,0,0,0,0],
					[0,0,1,0,1,1,1,0,1,0,0,0,0,0,0],[0,0,1,0,0,1,1,-1,0,0,0,0,0,0,0],
					[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
					[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
					[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
					]

		self.time = 0
		self.T_starting_time = 0
		self.WS_busy_duration = np.zeros(self.W_number)
		self.WS_resource = np.zeros(self.W_number * self.R_number)
		self.Complete_cycle_time = np.zeros(1)
		self.n_start_time = [] # execution start at specific location
		self.n_duration = []
		self.n_bay_start = [] # at which workstation
		self.n_job_id = []

		self.n_start_time_R = [] # resource assignment location
		self.n_duration_R = []
		self.n_bay_start_R = []
		self.n_job_id_R = []
		self.C_max = np.zeros(1)
		self.workstation_processing_task = np.zeros(self.W_number)
		# workstation status checking

	def twin_check(self, action_task_WS1, action_task_WS2):
		twin = 0
		if action_task_WS1 > 0:
			if action_task_WS1 == action_task_WS2:
				twin = 1
		return twin


	def step(self, action_task_WS1, action_resource_WS1, action_task_WS2, action_resource_WS2):

		actions = [action_task_WS1, action_task_WS2]
		C_max = self.C_max
		W = self.W_number
		T =self.T_number
		R =self.R_number

		W_state = self.state[:W]
		T_state = self.state[W:W+T]
		R_state =self.state[W+T:W+T+W*R]
		AssignedState = self.state[W+T+W*R:]        # final assigned state of task by respecting workstation, resource and precdience constraints
		tot_task_duration = self.Complete_cycle_time

		"""
		Resource constraints
		"""
		T_R_C = self.T_R_C                  # 	Resource constraints
		#T_P_C = self.T_P_C                  #   Task constraints

		# Processing variables
		Tasks_duration = self.Tasks_duration
		WS_busy_duration = self.WS_busy_duration # (6) [(1,0,0),(1,0,0)]
		WS_resource = self.WS_resource
		time = self.time

		good_resource_assignment_reward = 0
		bad_resource_assignment_reward = 0
		task_assignment_reward = 0

		a_task_W1 = self.A_T_W_1[action_task_WS1][:] # list of tasks (location) , action_task_WS1 is number
		a_task_W2 = self.A_T_W_2[action_task_WS2][:] # ....................................................

		a1 = np.concatenate([a_task_W1, a_task_W2])		# concatination of both task assignment  [1......30]

		a_resource_W1 = self.A_R_W_1[action_resource_WS1][:]
		a_resource_W2 = self.A_R_W_2[action_resource_WS2][:]

		a2 = np.concatenate([a_resource_W1, a_resource_W2])	# concatination of both resource assignment task [1....6]

		a_final_matrix = np.concatenate([a1, a2])               # final matrix of both task and resource assignment
		size_a = len(a_final_matrix)
		#print("the size of size_a:", a, size_a)
		AssignedStateTemp = AssignedState[0]
		# action assigns task?
		"""
		remaining task for execution for future assignment 
		"""
		remaing_task = 0

		for i in range(len(a1)):
			remaing_task += a_final_matrix[i]

		W = self.W_number
		T = self.T_number
		T_P_C = self.T_P_C


		unfeasible_resource = 0
		for i in range(W):
			for j in range(T):
				if a1[i*T+j] == 1:
					Required_R = [i for i, x in enumerate(T_R_C[j][:]) if x == 1]
					if R_state[i*R+Required_R[0]] == 0 and a2[i*R+Required_R[0]] == 0:
						unfeasible_resource = 1
						break

		# End of resource checking constraint
		"""
		Step: 2 Task assignment and  workstation busy duration 
		if condition satisfied if unfeasible_resource = 0 then we inter in task assignment loop 
		"""

		if unfeasible_resource== 0:
			for i in range(W):
				if W_state[i] == 1:
					WS_busy_duration[i]-=1
					tot_task_duration-=1
					if WS_busy_duration[i] == 0:
						W_state[i] = 0

			if remaing_task > 0:
				task_index = [i for i, x in enumerate(T_state) if x == 0]  # print("These task are still pending for execution", task_index)
				for i in range(W):
					for z in range(len(task_index)):
						if a_final_matrix[i*T+task_index[z]] == 1:      # we need following task to be assign
							T_state[task_index[z]] = 1
							W_state[i] = 1                               # task_assignment to workstation
							WS_busy_duration[i] = Tasks_duration[task_index[z]]
							AssignedState[0] += 1
							# Task execution start
							self.n_start_time.append(time)
							self.n_duration.append(Tasks_duration[task_index[z]])     # +1 is given by the release time
							self.n_bay_start.append(i)                                  # at which workstation task start execution
							self.n_job_id.append(str(task_index[z]+1))   				# job id

							Required_R = [t for t, x in enumerate(T_R_C[task_index[z]][:]) if x == 1]

							if a2[i*R+Required_R[0]] == 1:
								good_resource_assignment_reward += 1
								self.n_start_time_R.append(time)
								self.n_duration_R.append(0.9)    # +1 is given by the release time
								self.n_bay_start_R.append(i)
								self.n_job_id_R.append(str(Required_R[0]+1))
								a2[i*R+Required_R[0]] = 0

						else:
							WS_resource[i*R+Required_R[0]] -= 1
							if WS_resource[i*R+Required_R[0]] == 0:
								R_state[i*R+Required_R[0]]= 0

			"""
			Step3:Resource assignment to  workstation 
			"""
			for i in range(W):
				for k in range(R):
					if a2[i*R+k]== 1:
						if R_state[i*R+k] == 0:
							WS_resource[i*R+k] += 1
							R_state[i*R+k] = 1
							good_resource_assignment_reward += 1

							self.n_start_time_R.append(time)
							self.n_duration_R.append(1)    # +1 is given by the release time
							self.n_bay_start_R.append(i)   # resource assignment to workstation
							self.n_job_id_R.append(str(k+1))
						else:
							# WS_resource[i*R+k] += 1
							bad_resource_assignment_reward -= 1
							self.n_start_time_R.append(time)
							self.n_duration_R.append(0.9)
							self.n_bay_start_R.append(i)
							self.n_job_id_R.append(str(k+1))
		"""
		makspane calculation 
		"""
		if unfeasible_resource == 0:
			C_increment = 0
			C_temp_prec_action = 0
			for i in range(W):
				C_temp = 0
				if actions[i] == 0:
					C_temp = 1
				else:
					C_temp = Tasks_duration[actions[i]-1]+1
				max_index = np.argmax(WS_busy_duration)
				if C_temp <= WS_busy_duration[max_index]:
					C_temp = 0
				else:
					C_temp -= WS_busy_duration[max_index]
				if C_temp > C_temp_prec_action:
					C_increment = C_temp
					C_temp_prec_action = C_temp
			C_max[0] += C_increment

		total_reward = 0
		reward_t_r =0
		reward_r   = 0

		RewardResources = bad_resource_assignment_reward

		if unfeasible_resource==0:
			time +=1
		RewardTasks = 0
		if AssignedState[0] - AssignedStateTemp  > 0 and unfeasible_resource==0:
			reward	 = 1.5*(AssignedState[0] - AssignedStateTemp) + good_resource_assignment_reward
		else:
			reward= RewardResources

		#reward = RewardTasks  + RewardResources

		sumWS = 0
		for i in range(W):
			sumWS += W_state[i]

		if AssignedState[0] == T and sumWS == 0:
			done = True
			reward =  10*1/C_max[0]
		else:
			done = False

		self.state = np.concatenate([W_state, T_state, R_state, AssignedState])

		self.time = time
		return np.array(self.state), reward, done, {}

	def check_constraint(self, action_task_j):
			W = self.W_number
			T = self.T_number
			T_P_C = self.T_P_C
			a = self.A_T_W_1[action_task_j][:]
			T_state = self.state[W:W+T]
			unfeasible_request = 0
			Remaning_list_of_assigned_tasks = [i for i, x in enumerate(T_state) if x == 0]
			for j in range(len(Remaning_list_of_assigned_tasks)):
				if a[Remaning_list_of_assigned_tasks[j]]==1:
					task_constraint = [i for i, x in enumerate(T_P_C[Remaning_list_of_assigned_tasks[j]]) if x == -1]
					for t in range(len(task_constraint)):
						if T_state[task_constraint[t]] == 0 or self.workstation_processing_task[0]==(task_constraint[t]+1) or self.workstation_processing_task[1]==(task_constraint[t]+1):
							unfeasible_request = 1
							break
			return unfeasible_request

	def reset(self):
		self.state = np.zeros(self.W_number + self.T_number+self.W_number*self.R_number+1)
		return np.array(self.state)

	def reset_variable(self):
		self.WS_busy_duration = np.zeros(self.W_number)
		self.WS_resource =np.zeros(self.W_number*self.R_number)
		self.Complete_cycle_time = np.array([sum(self.Tasks_duration[:self.T_number])])
		self.time = self.T_starting_time
		self.C_max = np.zeros(1)
		self.n_start_time = []
		self.n_duration = []
		self.n_bay_start = []
		self.n_job_id = []
		self.n_start_time_R = []
		self.n_duration_R = []
		self.n_bay_start_R = []
		self.n_job_id_R = []
