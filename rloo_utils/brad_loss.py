import torch
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# gonna calculate my bradley terry loss
# bradley terry low-key sounds like a goated wide reciever from the 1990s
# if you told me he was on those troy aikman cowboy teams I would've believed you
def bradley_terry_loss(inputs_w, inputs_l, mask_w, mask_l, model):

    reward_w = model(input_ids=inputs_w.to(DEVICE), attn_mask=mask_w.to(DEVICE))
    reward_l = model(input_ids=inputs_l.to(DEVICE), attn_mask=mask_l.to(DEVICE))

    losses = F.logsigmoid(reward_w - reward_l)

    return -losses
