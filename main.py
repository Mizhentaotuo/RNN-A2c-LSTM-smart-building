# AI - brain
class Params():
    def __init__(self):
        self.lr = 0.0001
        self.gamma = 0.99
        self.tau = 1.
        self.seed = 1
        self.num_steps = 24
        self.num_steps_year = 365
        self.max_episode_length = 100
        self.env_name = 'BCVTB'
        self.hidden_layer = 64
        self.area = 2924.1
        self.file_path_txt = r'C:\Users\Lab PC\BCVTB\examples\ePlus85-schedule\data_need.txt'
        self.file_path_reward_info = r'C:\Users\Lab PC\BCVTB\examples\ePlus85-schedule\data_need.p' # pickle file
        self.file_path_states_actions = r'C:\Users\Lab PC\BCVTB\examples\ePlus85-schedule\states_actions.p' # pickle file
        self.file_path_reward_result = r'C:\Users\Lab PC\BCVTB\examples\ePlus85-schedule\reward_result.p' # pickle file
        self.file_path_prediction = r'C:\Users\Lab PC\BCVTB\examples\ePlus85-schedule\prediction.p' # pickle file
        self.file_path_norm = r'C:\Users\Lab PC\BCVTB\examples\ePlus85-schedule\data_minmax_scale.p' # pickle file
        self.file_path_model = r'C:\Users\Lab PC\BCVTB\examples\ePlus85-schedule\model.p' # pickle file
        self.file_path_shared_model = r'C:\Users\Lab PC\BCVTB\examples\ePlus85-schedule\shared_model.pth' # pickle file
        self.file_path_shared_model1 = r'C:\Users\Lab PC\BCVTB\examples\ePlus85-schedule\shared_model1.pth' # pickle file
        self.file_path_shared_model_test = r'C:\Users\Lab PC\BCVTB\examples\ePlus85-schedule\shared_model_test.pth' # pickle file
        self.file_path_model_test = r'C:\Users\Lab PC\BCVTB\examples\ePlus85-schedule\model_test.p' # pickle file
        self.file_path_optimizer = r'C:\Users\Lab PC\BCVTB\examples\ePlus85-schedule\optimizer.p' # pickle file
        self.comfort_parameters = ['comfort_ref', 'op_2C', 'op_0N', 'op_0S', 'op_0E', 'op_0C', 'op_1C', 'op_1S', 'op_1N', 'op_1W', 'op_2S', 'op_2N']
        self.cost_flex_parameters = ['Qh_zone', 'Qh_ven', 'Qc_zone', 'Qc_ven', 'P_level', 'Price', 'cost_ref']
        self.prediction_parameters = ['price_level', 'price', 'solar', 'temperature']
        self.output_space = np.array([1, 2, 3, 4, 5])
#        self.all_parameter_name = ['0Tout', '1sol', '2Qh_zone', '3Qh_ven', '4Qc_zone', '5Qc_ven', '6P_level', '7Price', '8comfort_ref', '9cost_ref',
#                                   '10op_2C', '11op_0N', '12op_0S', '13op_0E', '14op_0C', '15op_1C', '16op_1S', '17op_1N', '18op_1W', '19op_2S', '20op_2N',
#                                   '21P_2C', '22P_0N', '23P_0S', '24P_0E', '25P_0C', '26P_1C', '27P_1S', '28P_1N', '29P_1W', '30P_2S', '31P_2N',
#                                   '32L_2C', '33L_0N', '34L_0S', '35L_0E', '36L_0C', '37L_1C', '38L_1S', '39L_1N', '40L_1W', '41L_2S', '42L_2N',
#                                   '43EL_2C', '44EL_0N', '45EL_0S', '46EL_0E', '47EL_0C', '48EL_1C', '49EL_1S', '50EL_1N', '51EL_1W', '52EL_2S', '53EL_2N',
#                                   '54Zsol_2C', '55Zsol_0N', '56Zsol_0S', '57Zsol_0E', '58Zsol_0C', '59Zsol_1C', '60Zsol_1S', '61Zsol_1N', '62Zsol_1W', '63Zsol_2S', '64Zsol_2N',
#                                   '65Zinf_2C', '66Zinf_0N', '67Zinf_0S', '68Zinf_0E', '69Zinf_0C', '70Zinf_1C', '71Zinf_1S', '72Zinf_1N', '73Zinf_1W', '74Zinf_2S', '75Zinf_2N',
#                                   '76ZNV_2C', '77ZNV_0N', '78ZNV_0S', '79ZNV_0E', '80ZNV_0C', '81ZNV_1C', '82ZNV_1S', '83ZNV_1N', '84ZNV_1W', '85ZNV_2S', '86ZNV_2N',
#                                   '87Zh_2C', '88Zh_0N', '89Zh_0S', '90Zh_0E', '91Zh_0C', '92Zh_1C', '93Zh_1S', '94Zh_1N', '95Zh_1W', '96Zh_2S', '97Zh_2N',
#                                   '98Zc_2C', '99Zc_0N', '100Zc_0S', '101Zc_0E', '102Zc_0C', '103Zc_1C', '104Zc_1S', '105Zc_1N', '106Zc_1W', '107Zc_2S', '108Zc_2N',
#                                   '109Heating_setpoint', '110Cooling_setpoint', '111AI_control']
        
        ##'76Zven_2C', '77Zven_0N', '78Zven_0S', '79Zven_0E', '80Zven_0C', '81Zven_1C', '82Zven_1S', '83Zven_1N', '84Zven_1W', '85Zven_2S', '86Zven_2N',

class Values():
    def __init__(self):
        self.operative_temp_day = [] 
        self.cost_flex_day = [] 
        self.values = [] 
        self.log_probs = [] 
        self.entropies = [] 
        self.states = []
        self.actions = []

import sys
import numpy as np
#import pandas as pd
from random import sample
import pickle
import torch
import torch.nn.functional as F
from model import ActorCritic
from train import train
import my_optim
import copy

def rewardCalComfort(operative_temp_day): # average comfort index within one day considering all the zones 
    comfort_level = list(map(lambda y: list(map(lambda x: 3*(21<=x<=25.5) + 2*((25.5<x<=26) or (20<=x<21)) + 1*((26<x<=27) or (19<=x<20)), y)), operative_temp_day))
    comfort_index = 0
    comfort_ref = operative_temp_day[12][0]
    for i in range(params.num_steps):
        comfort_index += sum(comfort_level[i][1:]) * (operative_temp_day[i][0] > 0) / (3 * 11)
    reward_comfort = 0 if comfort_ref == 0 else ((comfort_index / 9 - comfort_ref) / comfort_ref)
    return 1 if reward_comfort >=0 else min(reward_comfort, -1)

def rewardCalCostFlexibility(cost_flex_day, area): # Qh_zones, Qh_vent, Qc_zones, Qc_vent, P_level, Price, cost_ref
    cost_F = 0
    cost_ref = sum(row[6] for row in cost_flex_day)
    Q_total = [sum(row[0:4]) for row in cost_flex_day]
    Q_lmh = 0
    for i in range(len(cost_flex_day)):
        cost_F += Q_total[i] * cost_flex_day[i][5] / (area * 1000 * 4) # total heating and cooling power [kWh/m2] 
        Q_lmh += Q_total[i] * (cost_flex_day[i][4] == 1) - 0.1 * Q_total[i] * (cost_flex_day[i][4] == 2) - Q_total[i] * (cost_flex_day[i][4] == 3) # (Q_low - 0.1 * Q_m - Q_high)
    if cost_ref == 0:
        reward_cost = -1 if cost_F > 0 else 0
    else:
        reward_cost = (cost_ref - cost_F) / cost_ref 
    reward_cost = max(min(reward_cost, 1), -1)
    reward_F = 0 if sum(Q_total) == 0 else Q_lmh / sum(Q_total)
    return reward_cost, reward_F

def predictionFlat(file_path_prediction, x):
    x = int(x)
    with open(file_path_prediction, "rb") as f: # collecting information for state
        df_prediction = pickle.load(f)
    prediction = df_prediction.iloc[:, :].values.tolist()
    if x <= 8736:
        data = prediction[(x) : (x + 24)]
    else:
        data = prediction[(x) : 8760] + prediction[0 : (24 - (8760 - x))]   
    price =[]
    solar = []
    temp = []
    for i in data:
        price.append(i[1])
        solar.append(i[2])
        temp.append(i[3])
    return price + solar + temp
        
def state_normalization(file_path_norm, state):
    data_norm = []
    with open(file_path_norm, "rb") as f:
        min_max = pickle.load(f)
    for i in range(len(state)):
        if state[i] > min_max[i][0]:
            data_norm.append(1)
        elif (state[i] < min_max[i][1]) or ((min_max[i][0] - min_max[i][1]) == 0):
            data_norm.append(0)
        else:
            data_norm.append((state[i] - min_max[i][1]) / (min_max[i][0] - min_max[i][1]))
    return data_norm

def model_save(file_path_model, cx, hx):
    # save model after training
    with open(file_path_model, "wb") as f:
        pickle.dump(cx, f)
        pickle.dump(hx, f)
        
def optimizer_save(file_path_optimizer, optimizer):
    # save model after training
    with open(file_path_optimizer, "wb") as f:
        pickle.dump(optimizer, f)

def save_checkpoint(filepath, model, optimizer, hx, cx):
    checkpoint = {'state_dict': model.state_dict(),
                  'optimizer' : optimizer.state_dict(),
                  'hx': hx,
                  'cx': cx}
    torch.save(checkpoint, filepath)

def save_daily_reward(file_path_reward_result, reward):
    with open(params.file_path_reward_result, "rb") as f:
        reward_result = pickle.load(f)
    reward_result.append(reward)
    with open(params.file_path_reward_result, "wb") as f:
        pickle.dump(reward_result, f)
### 

time_step_update = int(float(sys.argv[1])) / 3600

# initializing parameters
params = Params()

#sum_test = 0
training = 0
reward = 0
loss = 0
TD = 0
gae = 0
grad = 0
update_check = 0

# output action
x = sample([1, 2, 3, 4, 5], 1)

# read parameters from E+
all_parameter = list(map(float, sys.argv[2].strip('[]').split(';')))
length = len(all_parameter)
P_level = all_parameter[6]
price = all_parameter[7]
heating_setpoint = all_parameter[109]
cooling_setpoint = all_parameter[110]
AI_control = all_parameter[111]
operative_temp = [all_parameter[8]] + all_parameter[10:21]
cost_flex = all_parameter[2:8] + [all_parameter[9]]
state_num = all_parameter[10:109] + all_parameter[0:6] + [all_parameter[7]] + predictionFlat(params.file_path_prediction, (time_step_update) % 8760)
    
state = np.array(state_normalization(params.file_path_norm, state_num))
state = torch.from_numpy(state).float()
#state = torch.FloatTensor(state) # tensorizing the new state

if time_step_update == 0.0:
    values_info = Values()
    model = ActorCritic(len(state), params.output_space) # creating the model from the ActorCritic class
    optimizer = my_optim.SharedAdam(model.parameters(), lr=params.lr) # the optimizer is also shared because it acts on the shared model
    checkpoint = torch.load(params.file_path_shared_model)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()
    hx = checkpoint['hx'].data # we keep the old cell states, making sure they are in a torch variable
    cx = checkpoint['cx'].data # we keep the old hidden states, making sure they are in a torch variable
#    hx = torch.zeros(1, params.hidden_layer) # the cell states of the LSTM are reinitialized to zero
#    cx = torch.zeros(1, params.hidden_layer) # the hidden states of the LSTM are reinitialized to zero
#    model_save(params.file_path_model, hx, cx)
    model_test = copy.deepcopy(model)
    optimizer_test = copy.deepcopy(optimizer)
    save_checkpoint(params.file_path_shared_model, model, optimizer, hx, cx)
    save_checkpoint(params.file_path_shared_model_test, model_test, optimizer_test, hx, cx)
    #######
else:    
    # Load the lists back from the pickle file.
    with open(params.file_path_reward_info, "rb") as f:
        values_info = pickle.load(f)

    # load model
#    with open(params.file_path_model, "rb") as f:
#        cx = pickle.load(f)
#        hx = pickle.load(f)
    model = ActorCritic(len(state), params.output_space)
    optimizer = my_optim.SharedAdam(model.parameters(), lr=params.lr)
    checkpoint = torch.load(params.file_path_shared_model)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()
    hx = checkpoint['hx'].data # we keep the old cell states, making sure they are in a torch variable
    cx = checkpoint['cx'].data # we keep the old hidden states, making sure they are in a torch variable
   
value_now, action_values, _ = model((state.unsqueeze(0), (hx, cx))) # getting from the model the output V(S) of the critic, the output Q(S,A) of the actor, and the new hidden & cell states
prob = F.softmax(action_values, dim=1) # generating a distribution of probabilities of the Q-values according to the softmax: prob(a) = exp(prob(a))/sum_b(exp(prob(b)))
#log_prob = F.log_softmax(action_values, dim=1) # generating a distribution of log probabilities of the Q-values according to the log softmax: log_prob(a) = log(prob(a))
#entropy = -(log_prob * prob).sum(1) # H(p) = - sum_x p(x).log(p(x))
action = prob.multinomial(1).data # selecting an action by taking a random draw from the prob distribution
#log_prob = log_prob.gather(1, action) # getting the log prob associated to this selected action

#################
if time_step_update > 0.0:
    values_info.operative_temp_day.append(operative_temp) # used for comfort calculation 
    values_info.cost_flex_day.append(cost_flex) 

#################
x = action.numpy().tolist()[0][0] + 1
#################
    
if len(values_info.operative_temp_day) == params.num_steps:
    training = 1
    reward_comfort = rewardCalComfort(values_info.operative_temp_day)
    reward_cost, reward_F = rewardCalCostFlexibility(values_info.cost_flex_day, params.area)
    reward = (0.4 * reward_cost + 0.4 * reward_F + 0.2 * reward_comfort)
    rewards = (np.ones(params.num_steps) * reward).tolist() # storing the new observed reward
    save_daily_reward(params.file_path_reward_result, reward)
        
    for i in range(params.num_steps):
        value, action_values, (hx, cx) = model((values_info.states[i].unsqueeze(0), (hx, cx))) # getting from the model the output V(S) of the critic, the output Q(S,A) of the actor, and the new hidden & cell states
        prob = F.softmax(action_values, dim=1) # generating a distribution of probabilities of the Q-values according to the softmax: prob(a) = exp(prob(a))/sum_b(exp(prob(b)))
        log_prob = F.log_softmax(action_values, dim=1) # generating a distribution of log probabilities of the Q-values according to the log softmax: log_prob(a) = log(prob(a))
        entropy = -(log_prob * prob).sum(1) # H(p) = - sum_x p(x).log(p(x))
        action = values_info.actions[i].data # selecting an action by taking a random draw from the prob distribution
        log_prob = log_prob.gather(1, action) # getting the log prob associated to this selected action
        values_info.values.append(value)
        values_info.log_probs.append(log_prob)
        values_info.entropies.append(entropy)
        
    values_info.values.append(value_now) # storing the value V(S) of the state 
    loss, TD, gae, grad, update_check = train(params, optimizer, state, hx, cx, rewards, values_info.values, values_info.log_probs, values_info.entropies, model, value_now)
    model_save(params.file_path_model, cx, hx)
    save_checkpoint(params.file_path_shared_model, model, optimizer, hx, cx)
    values_info = Values() # clean and initialize the values
# Save states and actions for training
values_info.states.append(state)
values_info.actions.append(action)
 
# Save lists into a pickle file.
with open(params.file_path_reward_info, "wb") as f:
    pickle.dump(values_info, f)
with open(r'C:\Users\Lab PC\BCVTB\examples\ePlus85-schedule\ttttttttttttt.p', "wb") as f:
    pickle.dump(all_parameter, f)

print(time_step_update, x, AI_control, heating_setpoint, cooling_setpoint, reward, loss, update_check) # TD, gae, grad, length, 
#for p in model.parameters():
#    if p.grad is not None:
#        print(p.grad.data)
sys.exit(x)

#sum_test = sum(all_parameter)
#    model1 = ActorCritic(len(state), params.output_space) # creating the model from the ActorCritic class
#    optimizer1 = my_optim.SharedAdam(model1.parameters(), lr=params.lr) # the optimizer is also shared because it acts on the shared model
#    save_checkpoint(params.file_path_shared_model1, model1, optimizer1)

          
#values_info.values.append(value) # storing the value V(S) of the state
#values_info.log_probs.append(log_prob) # storing the log prob of the action
#values_info.entropies.append(entropy) # storing the computed entropy  
