from operator import itemgetter
from copy import deepcopy
import heapq
import numpy as np
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss

def get_embedding_weight(language_model):
    for module in language_model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == 30522: 
                return module.weight.detach()

def add_hooks(language_model):
    for module in language_model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == 30522: 
                module.weight.requires_grad = True
                module.register_full_backward_hook(extract_grad_hook)

extracted_grads = []           
def extract_grad_hook(module, grad_in, grad_out):
    extracted_grads.append(grad_out[0])
    
def get_average_grad(model, inputs, labels, trigger_token_ids, target_label=None, snli=False):
    ce_loss = CrossEntropyLoss()
    model.train()
    optimizer = optim.Adam(model.parameters())
    optimizer.zero_grad()
    global extracted_grads
    extracted_grads = []
    trigger_out = model(**inputs)
    loss = ce_loss(trigger_out.logits, labels)
    loss.backward()
    grads = extracted_grads[0].cpu()
    averaged_grad = torch.sum(grads, dim=0)
    averaged_grad = averaged_grad[1:len(trigger_token_ids)+1]
    return averaged_grad
    
"""
Contains different methods for attacking models. In particular, given the gradients for token
embeddings, it computes the optimal token replacements. This code runs on CPU.
"""
import torch
import numpy

def hotflip_attack(averaged_grad, embedding_matrix, trigger_token_ids,
                   increase_loss=False, num_candidates=1, target_tokens=None):
    """
    The "Hotflip" attack described in Equation (2) of the paper. This code is heavily inspired by
    the nice code of Paul Michel here https://github.com/pmichel31415/translate/blob/paul/
    pytorch_translate/research/adversarial/adversaries/brute_force_adversary.py

    This function takes in the model's average_grad over a batch of examples, the model's
    token embedding matrix, and the current trigger token IDs. It returns the top token
    candidates for each position.

    If increase_loss=True, then the attack reverses the sign of the gradient and tries to increase
    the loss (decrease the model's probability of the true class). For targeted attacks, you want
    to decrease the loss of the target class (increase_loss=False).
    """
    averaged_grad = averaged_grad.cpu()
    embedding_matrix = embedding_matrix.cpu()
    trigger_token_embeds = torch.nn.functional.embedding(torch.LongTensor(trigger_token_ids),
                                                         embedding_matrix).detach().unsqueeze(0)
    target_embeddings = embedding_matrix[target_tokens].detach()
    averaged_grad = averaged_grad.unsqueeze(0)
    gradient_dot_embedding_matrix = torch.einsum("bij,kj->bik",
                                                 (averaged_grad, target_embeddings))        
    if not increase_loss:
        gradient_dot_embedding_matrix *= -1    # lower versus increase the class probability.
    if num_candidates > 1: # get top k options
        _, best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_candidates, dim=2)
        actual_tokens = []
        for i in range(len(trigger_token_ids)):
            actual_tokens.append([target_tokens[idx] for idx in best_k_ids[0][i].tolist()])
        return actual_tokens
        # return best_k_ids.detach().cpu().numpy()[0]
    _, best_at_each_step = gradient_dot_embedding_matrix.max(2)
    return best_at_each_step[0].detach().cpu().numpy()

def hotflip_attack_g(averaged_grad, embedding_matrix, trigger_token_ids,
                   increase_loss=False, num_candidates=1):
    """
    The "Hotflip" attack described in Equation (2) of the paper. This code is heavily inspired by
    the nice code of Paul Michel here https://github.com/pmichel31415/translate/blob/paul/
    pytorch_translate/research/adversarial/adversaries/brute_force_adversary.py

    This function takes in the model's average_grad over a batch of examples, the model's
    token embedding matrix, and the current trigger token IDs. It returns the top token
    candidates for each position.

    If increase_loss=True, then the attack reverses the sign of the gradient and tries to increase
    the loss (decrease the model's probability of the true class). For targeted attacks, you want
    to decrease the loss of the target class (increase_loss=False).
    """
    averaged_grad = averaged_grad.cpu()
    embedding_matrix = embedding_matrix.cpu()
    trigger_token_embeds = torch.nn.functional.embedding(torch.LongTensor(trigger_token_ids),
                                                         embedding_matrix).detach().unsqueeze(0)
    averaged_grad = averaged_grad.unsqueeze(0)
    gradient_dot_embedding_matrix = torch.einsum("bij,kj->bik",
                                                 (averaged_grad, embedding_matrix))        
    if not increase_loss:
        gradient_dot_embedding_matrix *= -1    # lower versus increase the class probability.
    if num_candidates > 1: # get top k options
        _, best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_candidates, dim=2)
        return best_k_ids.detach().cpu().numpy()[0]
    _, best_at_each_step = gradient_dot_embedding_matrix.max(2)
    return best_at_each_step[0].detach().cpu().numpy()



def get_best_candidates(model, inputs, labels, trigger_token_ids, cand_trigger_token_ids, beam_size=1):
    """"
    Given the list of candidate trigger token ids (of number of trigger words by number of candidates
    per word), it finds the best new candidate trigger.
    This performs beam search in a left to right fashion.
    """
    # first round, no beams, just get the loss for each of the candidates in index 0.
    # (indices 1-end are just the old trigger)
    loss_per_candidate = get_loss_per_candidate(0, model, inputs, labels, trigger_token_ids,
                                                cand_trigger_token_ids)
    # maximize the loss
    top_candidates = heapq.nsmallest(beam_size, loss_per_candidate, key=itemgetter(1))

    # top_candidates now contains beam_size trigger sequences, each with a different 0th token
    for idx in range(1, len(trigger_token_ids)): # for all trigger tokens, skipping the 0th (we did it above)
        loss_per_candidate = []
        for cand, _ in top_candidates: # for all the beams, try all the candidates at idx
            loss_per_candidate.extend(get_loss_per_candidate(idx, model, inputs, labels, cand,
                                                             cand_trigger_token_ids))
        top_candidates = heapq.nsmallest(beam_size, loss_per_candidate, key=itemgetter(1))
    return max(top_candidates, key=itemgetter(1))[0]

def get_loss_per_candidate(index, model, inputs, labels, trigger_token_ids, cand_trigger_token_ids):
    """
    For a particular index, the function tries all of the candidate tokens for that index.
    The function returns a list containing the candidate triggers it tried, along with their loss.
    """
    # if isinstance(cand_trigger_token_ids[0], (numpy.int64, int)):
    #     print("Only 1 candidate for index detected, not searching")
    #     return trigger_token_ids
    loss_per_candidate = []
    # loss for the trigger without trying the candidates
    curr_loss, _ = evaluate_batch(model, inputs, labels)
    curr_loss = curr_loss.cpu().detach().numpy()
    loss_per_candidate.append((deepcopy(trigger_token_ids), curr_loss))
    for cand_id in range(len(cand_trigger_token_ids[0])):
        trigger_token_ids_one_replaced = deepcopy(trigger_token_ids) # copy trigger
        # print(trigger_token_ids)
        trigger_token_ids_one_replaced[index] = cand_trigger_token_ids[index][cand_id] # replace one token
        inputs['input_ids'][:, -len(trigger_token_ids):] = torch.tensor(trigger_token_ids_one_replaced).expand(len(labels), -1)
        loss, _ = evaluate_batch(model, inputs, labels)
        loss = loss.cpu().detach().numpy()
        loss_per_candidate.append((deepcopy(trigger_token_ids_one_replaced), loss))
    return loss_per_candidate

def evaluate_batch(model, inputs, labels):
    model.eval()
    ce_loss = CrossEntropyLoss()
    with torch.no_grad():
        out = model(**inputs)
    loss = ce_loss(out.logits, labels)
    acc = accuracy(out.logits, labels)
    return loss, acc

def convert_np_to_native(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_np_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_to_native(i) for i in obj]
    return obj

def accuracy(output, labels):
    _, preds = torch.max(output, dim=1)
    correct = torch.sum(preds == labels).item()
    return correct / len(labels)
