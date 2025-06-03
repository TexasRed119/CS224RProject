import torch
from tqdm import tqdm
import torch.nn.functional as F
from dpo_utils.full_tokenize import full_tokenize
from dpo_utils.dpo_loss import dpo_loss

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def sft_do_epoch(model, split, dataloader, tokenizer, optimizer, args, scheduler=None, curriculum_init=False):
    loss_item = 0
    all_losses = []

    if split == 'train':
        model.train()
    elif split == 'test' or curriculum_init:
        model.eval()

    for batch in tqdm(dataloader):
        query_and_completion = []
        for i in range(len(batch['query'])):
            query_and_completion.append(batch['query'][i] + ' ' + batch['completion'][i])
        query_and_completion = tokenizer(query_and_completion,
            padding=True,
            return_tensors='pt'
        )

        output = model(
            query_and_completion['input_ids'].to(DEVICE),
            attention_mask=query_and_completion['attention_mask'].to(DEVICE)
        )

        loss = 0
        for i in range(len(batch['query'])):
            query_ids = tokenizer.encode(batch['query'][i])
            completion_ids = tokenizer.encode(batch['completion'][i], return_tensors='pt')
            pred_start = len(query_ids)-1
            num_preds = len(completion_ids[0])
            pred_logits = output['logits'][i][pred_start:pred_start+num_preds]
            pred_probs = torch.nn.functional.softmax(pred_logits, dim=1)

            losses = F.nll_loss(torch.log(pred_probs), completion_ids.reshape(-1).to(DEVICE), reduction='none')
            loss += losses.sum()
            if curriculum_init:
                all_losses.append(losses.mean().item())  # use .mean to normalize for longer completions

        loss = loss / len(batch['query'])
        loss_item += loss.item()
        
        if split == 'train' and not curriculum_init:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.scheduler:
                scheduler.step()

    return loss_item, len(dataloader), all_losses


def dpo_do_epoch(model, ref_model, split, dataloader, tokenizer, optimizer, args, scheduler=None, curriculum_init=False):
    loss_item = 0
    all_losses = []

    if split == 'train':
        model.train()
    elif split == 'test' or curriculum_init:
        model.eval()

    for batch in tqdm(dataloader):
        inputs_w, inputs_l, mask_w, mask_l, prompt_mask_w, prompt_mask_l = full_tokenize(batch, tokenizer)

        losses = dpo_loss(inputs_w, inputs_l, mask_w=mask_w, mask_l=mask_l, model=model, ref_model=ref_model, beta=args.beta, prompt_mask_w=prompt_mask_w, prompt_mask_l=prompt_mask_l)

        loss = losses.mean()
        loss_item += loss.item()

        if curriculum_init:
            all_losses.append(loss.item())

        if split == 'train' and not curriculum_init:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.scheduler:
                scheduler.step()

    return loss_item, len(dataloader), all_losses

