import torch
import torch.nn.functional as F
from dpo_utils.dpo_loss import compute_log_prob
from countdown_eval import compute_score


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def rloo_loss(model, batch, all_outputs, x_and_y, prompt_mask, args):

    rewards = []
    for i in range(args.batch_size):
        example_rewards = []
        for k in range(args.k):
            ground_truth = {
                "target": batch["target"][i],
                "nums": batch["nums"][i]
            }
            reward = compute_score(all_outputs[i][k], ground_truth)
            example_rewards.append(reward)
        rewards.append(example_rewards)

    # x_and_y rn has shape (batch_size * k, seq_len) rn. we need rewards and log_prob to end up as (batch_size, k, seq_len)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
    
    '''
    # have no fear...tomas is here
    # for loop would be sooooo easy but matt's gonna give me shit
    # for calculating the baseline average across batch instead:
    # lets make a matrix with diagonal masked out...simple matrix multiplcation w rewwards will give us the leaving one out sums . then we just divide k - 1
    # current shape of rewards is gonna be (batch,1) multiple by our zero diagonal matrix of batch x batch
    diagonal_zero_matrix = torch.ones((args.batch_size, args.batch_size)) - torch.eye(args.batch_size)
    leave_one_out_sum = diagonal_zero_matrix @ rewards
    baselines = leave_one_out_sum / (args.batch_size - 1)
    # ^^ you might have to squeeze / unsqueeze depending on if rewards is (batch,) or (batch,1), but I think this will work as is
    '''
    eye = torch.eye(args.k, device=DEVICE)
    mask = (1.0 - eye)[None, :, :]
    masked_rewards = rewards[:, None, :] * mask
    baselines = masked_rewards.sum(dim=-1) / (args.k - 1)

    log_prob = compute_log_prob(model, x_and_y["input_ids"], x_and_y["attention_mask"], prompt_mask)
    log_probs = log_probs.view(args.batch_size, args.k)

    losses = -((rewards - baselines) * log_prob)

    return losses

