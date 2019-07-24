# Result Analysis
class Params():
    def __init__(self):
        self.lr = 0.0001
        self.gamma = 0.99
        self.tau = 1.
        self.seed = 1
        self.num_steps = 23
        self.num_steps_year = 365
        self.max_episode_length = 100
        self.env_name = 'BCVTB'
        self.hidden_layer = 64
        self.area = 2924.1
        self.file_path_txt = r'C:\Users\Lab PC\BCVTB\examples\ePlus85-schedule\data_need.txt'
        self.file_path_reward_info = r'C:\Users\Lab PC\BCVTB\examples\ePlus85-schedule\data_need.p' # pickle file
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

class Values():
    def __init__(self):
        self.operative_temp_day = [] 
        self.cost_flex_day = [] 
        self.values = [] 
        self.log_probs = [] 
        self.entropies = [] 

def rewardCalComfort(operative_temp_day, params): # average comfort index within one day considering all the zones 
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

def load_checkpoint(filepath):
#    checkpoint = torch.load(filepath)
#    model = checkpoint['model']
#    model.load_state_dict(checkpoint['state_dict'])
#    for parameter in model.parameters():
#        parameter.requires_grad = False
#    model.eval()
    #####################
    model = ActorCritic(len(state), params.output_space)
    optimizer = my_optim.SharedAdam(model.parameters(), lr=params.lr)
    checkpoint = torch.load(params.file_path_shared_model)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()
     
    model_test = ActorCritic(len(state), params.output_space)
    optimizer_test = my_optim.SharedAdam(model_test.parameters(), lr=params.lr)
    checkpoint = torch.load(params.file_path_shared_model_test)
    model_test.load_state_dict(checkpoint['state_dict'])
    optimizer_test.load_state_dict(checkpoint['optimizer'])
    model_test.eval()
    ###########################
    return model

def save_checkpoint(filepath, model, optimizer):
    checkpoint = {'state_dict': model.state_dict(),
                  'optimizer' : optimizer.state_dict()}
    torch.save(checkpoint, filepath)
    

import numpy as np
import pickle
import torch
import torch.nn.functional as F
from model import ActorCritic
import my_optim
import copy

time_step_update = 0
params = Params()


# read parameters from E+
with open(r'C:\Users\Lab PC\BCVTB\examples\ePlus85-schedule\ttttttttttttt.p', "rb") as f:
    all_parameter = pickle.load(f)
    state_num = pickle.load(f)
with open(params.file_path_reward_info, "rb") as f:
    values_info = pickle.load(f)
    
operative_temp = [all_parameter[8]] + all_parameter[10:21]
cost_flex = all_parameter[2:8] + [all_parameter[9]]
state_num = all_parameter[10:109] + all_parameter[0:6] + [all_parameter[7]] + predictionFlat(params.file_path_prediction, (time_step_update) % 8760)
    
state = np.array(state_normalization(params.file_path_norm, state_num))
state = torch.from_numpy(state).float()

cx = torch.zeros(1, params.hidden_layer) # the cell states of the LSTM are reinitialized to zero
hx = torch.zeros(1, params.hidden_layer) # the hidden states of the LSTM are reinitialized to zero

model = ActorCritic(178, params.output_space)
optimizer = my_optim.SharedAdam(model.parameters(), lr=params.lr)

value, action_values, (hx, cx) = model((state.unsqueeze(0), (hx, cx))) # getting from the model the output V(S) of the critic, the output Q(S,A) of the actor, and the new hidden & cell states
prob = F.softmax(action_values, dim=1) # generating a distribution of probabilities of the Q-values according to the softmax: prob(a) = exp(prob(a))/sum_b(exp(prob(b)))
log_prob = F.log_softmax(action_values, dim=1) # generating a distribution of log probabilities of the Q-values according to the log softmax: log_prob(a) = log(prob(a))
entropy = -(log_prob * prob).sum(1) # H(p) = - sum_x p(x).log(p(x))
action = prob.multinomial(1).data # selecting an action by taking a random draw from the prob distribution
log_prob = log_prob.gather(1, action) # getting the log prob associated to this selected action

#values_info.values.append(value) # storing the value V(S) of the state
reward_comfort = rewardCalComfort(values_info.operative_temp_day, params)
reward_cost, reward_F = rewardCalCostFlexibility(values_info.cost_flex_day, params.area)
reward = (0.4 * reward_cost + 0.4 * reward_F + 0.2 * reward_comfort)
rewards = (np.ones(params.num_steps) * reward).tolist() # storing the new observed reward

#with open(params.file_path_reward_result, "rb") as f:
#    reward_result = pickle.load(f)
#reward_result.append(reward)
#with open(params.file_path_reward_result, "wb") as f:
#    pickle.dump(reward_result, f)

model_before = copy.deepcopy(model)
before = list(model_before.parameters())    
###################
#train(params, optimizer, state, hx, cx, rewards, values_info.values, values_info.log_probs, values_info.entropies, model)

values = values_info.values   
log_probs = values_info.log_probs
entropies = values_info.entropies

value, _, _ = model((state.unsqueeze(0), (hx, cx))) # we initialize the cumulative reward with the value of the last shared state
R = value.data # we initialize the cumulative reward with the value of the last shared state
values_info.values.append(R) # storing the value V(S) of the last reached state S    
policy_loss = 0 # initializing the policy loss
value_loss = 0 # initializing the value loss
#R = R # making sure the cumulative reward R is a torch Variable
gae = torch.zeros(1, 1) # initializing the Generalized Advantage Estimation to 0
#for i in reversed(range(len(rewards))): # starting from the last exploration step and going back in time
#    R = params.gamma * R + rewards[i] # R = gamma*R + r_t = r_0 + gamma r_1 + gamma^2 * r_2 ... + gamma^(n-1)*r_(n-1) + gamma^nb_step * V(last_state)
#    advantage = R - values[i] # R is an estimator of Q at time t = i so advantage_i = Q_i - V(state_i) = R - value[i]
#    value_loss = value_loss + 0.5 * advantage.pow(2) # computing the value loss
#    TD = rewards[i] + params.gamma * values[i + 1].data - values[i].data # computing the temporal difference
#    gae = gae * params.gamma * params.tau + TD # gae = sum_i (gamma*tau)^i * TD(i) with gae_i = gae_(i+1)*gamma*tau + (r_i + gamma*V(state_i+1) - V(state_i))
#    policy_loss = policy_loss - log_probs[i] * gae - 0.01 * entropies[i] # computing the policy loss
    
#for i in reversed(range(len(rewards))): # starting from the last exploration step and going back in time
i = 1
R = params.gamma * R + rewards[i] # R = gamma*R + r_t = r_0 + gamma r_1 + gamma^2 * r_2 ... + gamma^(n-1)*r_(n-1) + gamma^nb_step * V(last_state)
advantage = R - value # R is an estimator of Q at time t = i so advantage_i = Q_i - V(state_i) = R - value[i]
value_loss = value_loss + 0.5 * advantage.pow(2) # computing the value loss
TD = rewards[i] + params.gamma * value.data - value.data # computing the temporal difference
gae = gae * params.gamma * params.tau + TD # gae = sum_i (gamma*tau)^i * TD(i) with gae_i = gae_(i+1)*gamma*tau + (r_i + gamma*V(state_i+1) - V(state_i))
policy_loss = policy_loss - log_prob * gae - 0.01 * entropy # computing the policy loss

################
optimizer.zero_grad() # initializing the optimizer
(policy_loss + value_loss).backward() # we give 2x more importance to the policy loss than the value loss because the policy loss is smaller
torch.nn.utils.clip_grad_norm_(model.parameters(), 40) # clamping the values of gradient between 0 and 40 to prevent the gradient from taking huge values and degenerating the algorithm
optimizer.step() # running the optimization step
#################
    
############
#optimizer_SGD = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
#optimizer_SGD.zero_grad()
##model.critic_linear.weight.retain_grad()
##model.actor_linear.weight.retain_grad()
##model.critic_linear.weight.requires_grad_()
##model.actor_linear.weight.requires_grad_()
#(policy_loss + value_loss).backward() # we give 2x more importance to the policy loss than the value loss because the policy loss is smaller
#optimizer_SGD.step()
#########
value_test = value.clone()      
with open(r'C:\Users\Lab PC\BCVTB\examples\ePlus85-schedule\data_need_test.p', "wb") as f:
    pickle.dump(value_test, f)
with open(r'C:\Users\Lab PC\BCVTB\examples\ePlus85-schedule\data_need_test.p', "rb") as f:
    value_test_test = pickle.load(f)

checkpoint_test = {'value_test': value_test}
torch.save(checkpoint_test, r'C:\Users\Lab PC\BCVTB\examples\ePlus85-schedule\shared_model_test_test.pth')  
checkpoint_test = torch.load(r'C:\Users\Lab PC\BCVTB\examples\ePlus85-schedule\shared_model_test_test.pth')
value_test_test = checkpoint_test['value_test']
#return policy_loss + 0.5 * value_loss
################################
after = list(model.parameters())
for i in range(len(before)):
    print(torch.equal(before[i].data, after[i].data))
for name, param in model.named_parameters():
    if param.requires_grad:
        print (name, param.data, param.grad, param.is_leaf)

#states[0].unsqueeze(0)        
for p in model.parameters():
    if p.grad is not None:
        print(p.grad.data)
#x = torch.tensor([[1., -1.], [1., 1.]], requires_grad=True)
#out = x.pow(2).sum()
#out.backward()
#x.grad
        
#########################################
model = ActorCritic(178, params.output_space)
optimizer = my_optim.SharedAdam(model.parameters(), lr=params.lr)
checkpoint = torch.load(params.file_path_shared_model)
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
model.eval()
model_parameter = list(model.parameters())   
 
model_test = ActorCritic(178, params.output_space)
optimizer_test = my_optim.SharedAdam(model_test.parameters(), lr=params.lr)
checkpoint_test = torch.load(params.file_path_shared_model_test)
model_test.load_state_dict(checkpoint_test['state_dict'])
optimizer_test.load_state_dict(checkpoint_test['optimizer'])
model_test.eval()
model_test_parameter = list(model_test.parameters()) 
#
for i in range(len(model_parameter)):
    print(torch.equal(model_parameter[i].data, model_test_parameter[i].data), model_parameter)
#########################################
reward_year = []
with open(params.file_path_reward_result, "rb") as f:
    reward_result = pickle.load(f)
for i in range(int(len(reward_result)/365)):
    reward_year.append(np.mean(reward_result[(i)*365:(i+1)*365]))
import matplotlib.pyplot as plt
plt.plot(reward_year)
plt.ylabel('reward')
plt.show()   

with open(params.file_path_reward_result, "rb") as f:
    reward_result = pickle.load(f)  

with open(r'C:\Users\Lab PC\BCVTB\examples\ePlus85-schedule\reward_result - Copy.p', "rb") as f:
    reward_result_old = pickle.load(f)      
len(reward_result_old)-len(reward_result_old)%365   
reward_new = reward_result_old[0:11680] + reward_result[:]
reward_result = reward_new

with open(params.file_path_reward_result, "wb") as f:
        pickle.dump(reward_result, f)
with open(r'C:\Users\Lab PC\BCVTB\examples\ePlus85-schedule\reward_result - Copy.p', "wb") as f:
        pickle.dump(reward_result, f)
# =============================================================================
# for i in prediction:
#     price.append(i[1])
#     solar.append(i[2])
#     temp.append(i[3])
# min_max.append([max(temp), min(temp)])   
# min_max.append([max(solar), min(solar)])
# 
# 
# min_max.append([max(price), min(price)])
# for i in range(24):
#     min_max.append([max(price), min(price)])
# for i in range(24):
#     min_max.append([max(solar), min(solar)])
# for i in range(24):
#     min_max.append([max(temp), min(temp)])
# =============================================================================
# =============================================================================
# # data_comfort = pd.read_csv(params.file_path_txt) 
# #    df_operative_temp_day = pd.DataFrame(np.array(operative_temp_day), columns=params.comfort_parameters)
# #    df_cost_flex_day = pd.DataFrame(np.array(cost_flex_day), columns=params.cost_flex_parameters)
# #df_operative_temp_day.to_csv(params.file_path_txt, index=False)
# =============================================================================
#################
#lstm = nn.LSTMCell(len(state), 256)
#inputs, (hx, cx) = inputs
#hx, cx = lstm(state.unsqueeze(0), (hx, cx))
#################
#with open(params.file_path_states_actions, "rb") as f:
#    states = pickle.load(f)
#    actions = pickle.load(f)
#with open(params.file_path_states_actions, "wb") as f:
#    pickle.dump(states, f)
#    pickle.dump(actions, f)
