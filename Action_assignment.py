#             cd Documents/ScriptPython
#             python ActionsSet_MultiAgentWS.py
import numpy as np
import random

def task_assignment_action(T):
    k = T
    job_assignment_action = []
    null_action = np.zeros(k)
    Identity = np.identity(k)
    job_assignment_action.append(null_action)
    for i in range(k):
        job_assignment_action.append(Identity[i][:])
    return job_assignment_action

def Resource_assignmnet_action(R):
    t = R
    Resource_assignment = []
    null_action = np.zeros(t)
    Identity = np.identity(t)
    Resource_assignment.append(null_action)
    for i in range(t):
        Resource_assignment.append(Identity[i][:])
    return Resource_assignment

def Task_Resource_constraint_generator(R,T):
    Identity = np.identity(R)
    T_R_C = []
    for i in range(T):
        T_R_C.append(Identity[random.randint(0,R-1)][:])
    #print('TRC :',T_R_C)
    return T_R_C
