import gym
import numpy as np
from gym import spaces
from Task_Res_Action_assignment import Action_assignment, Resource_assignmnet_action
class Assembly_line_Env(gym.Env):
	def __init__(self):
		self.WorkstationsNumber = 2
		self.TasksNumber = 15
		self.ResourcesNumber = 3
		self.ActionsSet_Task_W1 = Action_assignment(self.TasksNumber)
		self.ActionsSet_Resource_W1 = Resource_assignmnet_action(self.ResourcesNumber)
		self.ActionsSet_Task_W2 = Action_assignment(self.TasksNumber)
		self.ActionsSet_Resource_W2 = Resource_assignmnet_action(self.ResourcesNumber)
		high = np.full(self.WorkstationsNumber + self.TasksNumber + self.WorkstationsNumber*self.ResourcesNumber+1, np.finfo(np.float32).max)

		# Number of actions in our environment
		self.ActionSpace_task_number = spaces.Discrete(self.TasksNumber+1)
		self.ActionSpace_resource_number = spaces.Discrete(self.ResourcesNumber+1)

		# Observation space in our case
		self.observation_space = spaces.Box(-high, high, dtype=np.float32)
		self.state = None
		self.steps_beyond_done = None
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
		self.time_init = 0
		self.WS_busy_duration = np.zeros(self.WorkstationsNumber)
		self.WS_resource = np.zeros(self.WorkstationsNumber * self.ResourcesNumber)
		self.total_Tasks_duration = np.zeros(1)
		self.n_start_time = [] # execution start at specific location
		self.n_duration = []
		self.n_bay_start = [] # at which workstation
		self.n_job_id = []
		self.n_start_time_R = [] # resource assignment location
		self.n_duration_R = []
		self.n_bay_start_R = []
		self.n_job_id_R = []
		self.C_max = np.zeros(1)
		self.workstation_processing_task = np.zeros(self.WorkstationsNumber)
		# workstation status checking

	def twin_check(self, action_task_WS1, action_task_WS2):
		twin = 0
		if action_task_WS1 > 0:
			if action_task_WS1 == action_task_WS2:
				#print("work_station_1 and 2 requesting same task")
				twin = 1
		return twin

	def step(self, action_task_WS1, action_resource_WS1, action_task_WS2, action_resource_WS2):
		"""
		:param action_task_WS1:
		:param action_resource_WS1:
		:param action_task_WS2:
		:param action_resource_WS2:
		:return:
		"""

		actions = [action_task_WS1, action_task_WS2]
		# print("Available actions:", actions)

		C_max = self.C_max
		W = self.WorkstationsNumber
		T =self.TasksNumber
		R =self.ResourcesNumber

		WorkstationsState = self.state[:W]
		TasksState = self.state[W:W+T]
		ResourcesState =self.state[W+T:W+T+W*R]
		AssignedState = self.state[W+T+W*R:]        # final assigned state of task by respecting workstation, resource and precdience constraints
		tot_task_duration = self.total_Tasks_duration

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

		"""
		Resource assignment reward 
		task assignment reward
		comments: Both task and resource are learning skills for agent 
		task concatination
		"""

		good_resource_assignment_reward = 0
		bad_resource_assignment_reward = 0
		task_assignment_reward = 0

		a_task_W1 = self.ActionsSet_Task_W1[action_task_WS1][:] # list of tasks (location) , action_task_WS1 is number
		a_task_W2 = self.ActionsSet_Task_W1[action_task_WS2][:] # ....................................................

		a1 = np.concatenate([a_task_W1, a_task_W2])		# concatination of both task assignment  [1......30]

		a_resource_W1 = self.ActionsSet_Resource_W1[action_resource_WS1][:]
		a_resource_W2 = self.ActionsSet_Resource_W2[action_resource_WS2][:]

		a2 = np.concatenate([a_resource_W1, a_resource_W2])	# concatination of both resource assignment task [1....6]

		"""
		a_1: list of all available actions at starting point 
		a_2: List of assigned resource 
		"""
		"""
		a1: joint action
		a2: for resources (r1,r2) assignment action joint action
		a_final_matrix of actions : is final action for execution respecting all constraint , resource , presidence and workstation availabily
		if a is feasible this action is feasible in our case 
		"""
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

		W = self.WorkstationsNumber
		T = self.TasksNumber
		T_P_C = self.T_P_C

		"""
		Step: 1
		Need to be debug 
		Constraint check: Resource status, currently assigned task and then we assign an other task if that available
		Resource constraint checking:
		first we check what type of  resource is required for task execution e.g 1,2,3 
		"""
		unfeasible_resource = 0
		for i in range(W):
			for j in range(T):
				if a1[i*T+j] == 1:
					Required_R = [i for i, x in enumerate(T_R_C[j][:]) if x == 1]
					if ResourcesState[i*R+Required_R[0]] == 0 and a2[i*R+Required_R[0]] == 0:
						unfeasible_resource = 1
						break
				if unfeasible_resource == 1:
					break
		# End of resource checking constraint


		"""
		Step: 2 Task assignment and  workstation busy duration 
		"""
		if unfeasible_resource == 0:
			for i in range(W):
				if WorkstationsState[i] == 1:  # workstation is working on task
					WS_busy_duration[i]-=1 	   # we assign the task duartion to workstation busy duration
					tot_task_duration-=1	   # we reduce the by 1
					if WS_busy_duration[i] == 0:
						WorkstationsState[i] = 0

			if remaing_task > 0:
				task_index = [i for i, x in enumerate(TasksState) if x == 0] # print("These task are still pending for execution", task_index)
				for i in range(W):
					for z in range(len(task_index)):
						if a_final_matrix[i*T+task_index[z]] == 1:
							TasksState[task_index[z]] = 1
							WorkstationsState[i] = 1                               # task_assignment to workstation
							WS_busy_duration[i] = Tasks_duration[task_index[z]]
							AssignedState[0] += 1
							# Task execution start
							self.n_start_time.append(time)
							self.n_duration.append(Tasks_duration[task_index[z]]+1)     # +1 is given by the release time
							self.n_bay_start.append(i)                                  # at which workstation task start execution
							self.n_job_id.append(str(task_index[z]+1))   				# job id
							#print('Task # ', task_index[z]+1 ,'which has a duration of ',Tasks_duration[task_index[z]], ' has been assigned to WS : ',i+1, " at the following state", AssignedState)
							# decrease in resources
							Required_R = [t for t, x in enumerate(T_R_C[task_index[z]][:]) if x == 1] # the resource constraint we can only process task if specific resource is available
							#print( there are constraints on resources: task ', task_index [z] + 1,' requires resource ', Required_R [0] + 1,' to be processed on the WS: ', i + 1)
							if a2[i*R+Required_R[0]] == 1:
								good_resource_assignment_reward += 1
								# print('the action assign the resource simultaneously the action VERY GOOD ')
								self.n_start_time_R.append(time)
								self.n_duration_R.append(0.9)    # +1 is given by the release time
								self.n_bay_start_R.append(i)
								self.n_job_id_R.append(str(Required_R[0]+1))
								a2[i*R+Required_R[0]] = 0
							else:
								WS_resource[i*R+Required_R[0]] -= 1
								if WS_resource[i*R+Required_R[0]] == 0:
									ResourcesState[i*R+Required_R[0]]= 0
									#print('the resources have run out', Required_R[0]+1,'in the WS ', i+1)
					# print('Task # ', task_index[z]+1 ,'which has a duration of ',Tasks_duration[task_index[z]], ' has been assigned to WS : ',i+1, " at the following state", AssignedState)
		#########################################################################################################################################
			# Resource assignment to workstations after action a2[]

			"""
			Step3:Resource assignment to  workstation 
			"""
			for i in range(W):
				for k in range(R):
					if a2[i*R+k]== 1:
						if ResourcesState[i*R+k] == 0:
							WS_resource[i*R+k] += 1
							# print("'the resource', k + 1, 'is assigned to the WS', i + 1")
							ResourcesState[i*R+k] = 1
							good_resource_assignment_reward += 1
							# print("'the resource', k + 1, 'in the WS', i + 1, 'changes from missing to present GOOD", good_resource_assignment_reward)
							# Resource assignment to workstation
							self.n_start_time_R.append(time)
							self.n_duration_R.append(1)    # +1 is given by the release time
							self.n_bay_start_R.append(i)   # resource assignment to workstation
							self.n_job_id_R.append(str(k+1))
						else:
							# WS_resource[i*R+k] += 1
							bad_resource_assignment_reward -= 1
							# print('the resources ',k+1,' is assigned to the WS ',i+1,' is bad ')
							self.n_start_time_R.append(time)
							self.n_duration_R.append(0.9)    # +1 is given by the release time
							self.n_bay_start_R.append(i)
							self.n_job_id_R.append(str(k+1))
		"""
		makspane calculation 
		"""
		if unfeasible_resource ==0:
			C_increment = 0
			C_temp_prec_action = 0
			for i in range(W):
				C_temp = 0
				if actions[i] == 0:
					C_temp = 1
				else:
					C_temp = Tasks_duration[actions[i]-1]+1
					#print("maxmum task duaration:", C_temp)
				max_index = np.argmax(WS_busy_duration)
				if C_temp <= WS_busy_duration[max_index]:
					C_temp = 0
				else:
					C_temp -= WS_busy_duration[max_index]
					print(C_temp_prec_action)
				if C_temp > C_temp_prec_action:
					C_increment = C_temp
					C_temp_prec_action = C_temp
			C_max[0] += C_increment
			# the value calculated during each time step
			#print("Cmax =", C_max[0])

		# Reward calculation

		total_reward = 0
		RewardResources =  good_resource_assignment_reward + bad_resource_assignment_reward

		if unfeasible_resource ==0:
			time +=1
		RewardTasks = 0
		if AssignedState[0] - AssignedStateTemp  > 0:
			RewardTasks = 5 *(AssignedState[0] - AssignedStateTemp)
			#RewardTasks = 1
		reward = RewardTasks  + RewardResources
		sumWS = 0
		for i in range(W):
			sumWS += WorkstationsState[i]

		if AssignedState[0] == T and sumWS == 0:
			done = True
			#reward = RewardTasks  + RewardResources

			#reward =  10 * 1 /C_max[0]
		else:
			done = False

		self.state = np.concatenate([WorkstationsState, TasksState, ResourcesState, AssignedState])

		self.time = time

		return np.array(self.state), reward, done, {}

	def check_constraint(self, action_task_j):
			W = self.WorkstationsNumber
			T = self.TasksNumber
			T_P_C = self.T_P_C
			a = self.ActionsSet_Task_W1[action_task_j][:]
			#print('L AZIONE è : ',a)
			TasksState = self.state[W:W+T]
			unfeasible_request = 0
			lista_task_da_assegnare = [i for i, x in enumerate(TasksState) if x == 0]
			#print('lista delle posizioni dei task da asssengare sono : ',lista_task_da_assegnare, '(quindi aggiungi +1 per il Task)')
			for j in range(len(lista_task_da_assegnare)):      #per tutti i task ancora da assegnare
				if a[lista_task_da_assegnare[j]]==1:      # sta richiedendo il task j = lista_task_da_assegnare[j]
					#print(' la linea di T_P_C considerata è :',T_P_C[lista_task_da_assegnare[j]])
					task_constraint = [i for i, x in enumerate(T_P_C[lista_task_da_assegnare[j]]) if x == -1]  # task_constraint contiene le postazioni in cui T_R_C alla linea del task assegnato è -1
					#print('sta richiedendo il task',lista_task_da_assegnare[j]+1,' task_constraint : ',task_constraint)
					for t in range(len(task_constraint)):
						#print('verifico se l assegnazione del task', lista_task_da_assegnare[j]+1 ,' genera conflitti con la constraint di precedenza con il task',task_constraint[t]+1)
						#print(' ------->',task_constraint[t]+1)
						if TasksState[task_constraint[t]] == 0 or self.workstation_processing_task[0]==(task_constraint[t]+1) or self.workstation_processing_task[1]==(task_constraint[t]+1): # <--- se è in processamento (stato task è passato ad 1)
							#print('unfeasible !!  Task ',lista_task_da_assegnare[j]+1,' can not be requested before the task ',task_constraint[t]+1,'is processed')
							unfeasible_request = 1
							break
					if unfeasible_request ==1:
						break
			return unfeasible_request

	"""
	Reset function of Reinforcement Learning part after end of each episode.
	"""

	def reset(self):
		self.state = np.zeros(self.WorkstationsNumber + self.TasksNumber+self.WorkstationsNumber*self.ResourcesNumber+1)
		return np.array(self.state)

	def reset_variable(self):
		self.WS_busy_duration = np.zeros(self.WorkstationsNumber)
		self.WS_resource =np.zeros(self.WorkstationsNumber*self.ResourcesNumber)
		self.total_Tasks_duration = np.array([sum(self.Tasks_duration[:self.TasksNumber])])
		self.time = self.time_init
		self.C_max = np.zeros(1)
		self.n_start_time = []
		self.n_duration = []
		self.n_bay_start = []
		self.n_job_id = []
		self.n_start_time_R = []
		self.n_duration_R = []
		self.n_bay_start_R = []
		self.n_job_id_R = []

##################### Comment section #################

	"""
		unfeasible_resource = 0
		for i in range(W):
			for j in range(T):
				if a1[i*T+j] == 1: # if i assign task j to the WS i are there the resources?
					Required_R = [i for i, x in enumerate(T_R_C[j][:]) if x == 1]
					if ResourcesState[i*R+Required_R[0]]== 0 and a2[i*R+Required_R[0]]==0:
						unfeasible_resource = 1
						break
						
						
# total reward assigned to agent during task and resources assignment and resources after relese of the resources.
		#Reward calculation section
		# Resource assignment
		#print('bad resource assignment is ', bad_resource_assignment_reward)						
	"""
