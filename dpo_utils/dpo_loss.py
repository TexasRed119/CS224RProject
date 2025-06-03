import torch
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# function to calcualte the log_probs and then use the prompt mask to mask the prompt before calcualting loss
# we calculate all 4 log probs in equation using this
def compute_log_prob(model, inputs, attention_mask, prompt_mask):
    outputs = model(input_ids=inputs.to(DEVICE), attention_mask=attention_mask.to(DEVICE))
    logits = outputs.logits

    pred_probs = F.softmax(logits, dim=-1)

    # model is predicting next token, so we need to shift everything by one to compare to ground truth (which is the input in our case)
    # right now we have shape of (batch, sequence, vocab), so adjust over sequence dimension
    pred_probs = pred_probs[:, :-1, :]  # don't need last token prediction
    target_ids = inputs[:, 1:]  # our ground truth labels
    prompt_mask = prompt_mask[:, 1:]
    attention_mask = attention_mask[:, 1:]

    # log-probabilities of the target tokens
    target_preds = torch.gather(pred_probs.to(DEVICE), 2, target_ids.unsqueeze(-1).to(DEVICE)).squeeze(-1)

    log_probs = torch.log(target_preds)

    # we don't care about the prediction loss for the prompts, only the responses
    log_probs = log_probs * prompt_mask.to(DEVICE) * attention_mask.to(DEVICE)

    return log_probs.sum(dim=-1)  # need to sum over all of the tokens in the response

# apparently common values for beta are 0.1, 0.5, possible range of 0.01 -> 10. High values will overfit, low values will underfit
# 0.1 beta with 10 epochs might be similar to 1 epoch with 10 beta
def dpo_loss(inputs_w, inputs_l, mask_w, mask_l, model, ref_model, beta, prompt_mask_w, prompt_mask_l):

    logprob_model_w = compute_log_prob(model, inputs_w, mask_w, prompt_mask_w)
    logprob_model_l = compute_log_prob(model, inputs_l, mask_l, prompt_mask_l)

    # reference model is frozen, so don't calculate gradients
    with torch.no_grad():
        logprob_ref_w = compute_log_prob(ref_model, inputs_w, mask_w, prompt_mask_w)
        logprob_ref_l = compute_log_prob(ref_model, inputs_l, mask_l, prompt_mask_l)

    # changed the fomula a little, but mathematically the same. was easier to take 4 log_probs and then subtract, instead doing division and then log
    logits = beta * ((logprob_model_w - logprob_ref_w) - (logprob_model_l -  logprob_ref_l))
    losses = -F.logsigmoid(logits)
    return losses

# def wolff_log_prob(batch, model, tokenizer, completion_type):
#     query_and_completion = []
#     for i in range(len(batch['prompt'])):
#         query_and_completion.append(batch['prompt'][i] + ' ' + batch[completion_type][i])
#     query_and_completion = tokenizer(query_and_completion,
#         padding=True,
#         return_tensors='pt'
#     )

#     output = model(
#         query_and_completion['input_ids'].to(DEVICE),
#         attention_mask=query_and_completion['attention_mask'].to(DEVICE)
#     )

#     batch_probs = []
#     for i in range(len(batch['prompt'])):
#         query_ids = tokenizer.encode(batch['prompt'][i])
#         completion_ids = tokenizer.encode(batch[completion_type][i], return_tensors='pt')
#         pred_start = len(query_ids)-1
#         num_preds = len(completion_ids[0])
#         pred_logits = output['logits'][i][pred_start:pred_start+num_preds]
#         pred_probs = torch.nn.functional.softmax(pred_logits, dim=1)
#         pred_completion_probs = torch.gather(pred_probs, 1, completion_ids.reshape((-1, 1)).to(DEVICE))
#         log_probs = torch.log(pred_completion_probs)
#         batch_probs.append(log_probs.sum())

#     return torch.Tensor(batch_probs)

# def wolff_dpo_loss(batch, model, ref_model, beta, tokenizer):

#     logprob_model_w = wolff_log_prob(batch, model, tokenizer, 'chosen')
#     logprob_model_l = wolff_log_prob(batch, model, tokenizer, 'rejected')

#     # reference model is frozen, so don't calculate gradients
#     with torch.no_grad():
#         logprob_ref_w = wolff_log_prob(batch, ref_model, tokenizer, 'chosen')
#         logprob_ref_l = wolff_log_prob(batch, ref_model, tokenizer, 'rejected')

#     # changed the fomula a little, but mathematically the same. was easier to take 4 log_probs and then subtract, instead doing division and then log
#     logits = beta * ((logprob_model_w - logprob_ref_w) - (logprob_model_l -  logprob_ref_l))
#     losses = -F.logsigmoid(logits)
#     return losses
