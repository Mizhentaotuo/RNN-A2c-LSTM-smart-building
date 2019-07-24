# Training the AI

import torch
import copy
def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad
        
def train(params, optimizer, state, hx, cx, rewards, values, log_probs, entropies, model, value_now):
#    value, _, _ = model((state.unsqueeze(0), (hx, cx))) # we initialize the cumulative reward with the value of the last shared state
    R = value_now.data # we initialize the cumulative reward with the value of the last shared state
#    values.append(R) # storing the value V(S) of the last reached state S    
    policy_loss = 0 # initializing the policy loss
    value_loss = 0 # initializing the value loss
#    R = Variable(R) # making sure the cumulative reward R is a torch Variable
    gae = torch.zeros(1, 1) # initializing the Generalized Advantage Estimation to 0
    for i in reversed(range(len(rewards))): # starting from the last exploration step and going back in time
        R = params.gamma * R + rewards[i] # R = gamma*R + r_t = r_0 + gamma r_1 + gamma^2 * r_2 ... + gamma^(n-1)*r_(n-1) + gamma^nb_step * V(last_state)
        advantage = R - values[i] # R is an estimator of Q at time t = i so advantage_i = Q_i - V(state_i) = R - value[i]
        value_loss = value_loss + 0.5 * advantage.pow(2) # computing the value loss
        TD = rewards[i] + params.gamma * values[i + 1].data - values[i].data # computing the temporal difference
        gae = gae * params.gamma * params.tau + TD # gae = sum_i (gamma*tau)^i * TD(i) with gae_i = gae_(i+1)*gamma*tau + (r_i + gamma*V(state_i+1) - V(state_i))
        policy_loss = policy_loss - log_probs[i] * gae - 0.01 * entropies[i] # computing the policy loss
    optimizer.zero_grad() # initializing the optimizer
    (policy_loss + value_loss).backward() # we give 2x more importance to the policy loss than the value loss because the policy loss is smaller
#    grad = []
#    for i in len(model.parameters()):
#        grad[i] = model.parameters()[i].grad.data
    model1 = copy.deepcopy(model)
    before = list(model1.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 40) # clamping the values of gradient between 0 and 40 to prevent the gradient from taking huge values and degenerating the algorithm
#    ensure_shared_grads(model, shared_model)
    optimizer.step() # running the optimization step
    after = list(model.parameters())
    update_check = 1
    for i in range(len(before)):
        update_check = 0 if torch.equal(before[i].data, after[i].data) == True else update_check        
    return policy_loss + value_loss, TD, gae, model.actor_linear.weight.grad, update_check

