def append_sample (self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # target nn updating after number of time step
def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.update_target_model()

def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform_(m.weight)

    # espsilon its looks like policy or action selection

    # functions for resource and task selection.
def get_action(self, state,  epsilon_custom):
        if np.random.rand() <= epsilon_custom:
            return random.randrange(self.action_size)
        else:
            state = torch.from_numpy(state)
            state = Variable(state).float().cpu()
            q_value = self.model(state)
            _, action = torch.max(q_value, 1)
            return int(action)
    # function for action selection for task

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
        return action , rand_action
    """
    In this function we check the pre constraint and wrorkstation status
    """

    # function for checking the pr-constraint constraints
    # we call it as nested function
    # names are very confusing
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
                    list_masked_task.append(action_task_WS-1)
                    num_unfeasible_taken_action += 1
                    print("final action selected:", action_task_WS, "Unfeasible action at following iteration:", num_unfeasible_taken_action, "list of masked task:", list_masked_task)
        return action_task_WS, list_masked_task




