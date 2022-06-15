#             cd Documents/ScriptPython
#             python ActionsSet_MultiAgentWS.py
import numpy as np
import random

def ActionsSet_Task(T):
    k = T
    actionSet_Task = []
    null_action = np.zeros(k)
    Identity = np.identity(k)
    actionSet_Task.append(null_action)
    for i in range(k):
        actionSet_Task.append(Identity[i][:])
    return actionSet_Task


def ActionsSet_Resource(R):
    t = R
    actionSet_Resource = []
    null_action = np.zeros(t)
    Identity = np.identity(t)
    actionSet_Resource.append(null_action)
    for i in range(t):
        actionSet_Resource.append(Identity[i][:])
    return actionSet_Resource

def Task_Resource_constraint_generator(R,T):
    Identity = np.identity(R)
    T_R_C = []
    for i in range(T):
        T_R_C.append(Identity[random.randint(0,R-1)][:])
    #print('TRC :',T_R_C)
    return T_R_C
