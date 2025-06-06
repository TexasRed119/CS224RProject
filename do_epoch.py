import torch
from tqdm import tqdm
import torch.nn.functional as F
from dpo_utils.full_tokenize import full_tokenize
from dpo_utils.dpo_loss import dpo_loss
from rloo_utils.brad_loss import bradley_terry_loss
import numpy as np

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
        # test = wolff_dpo_loss(batch, model, ref_model, args.beta, tokenizer)

        loss = losses.mean()
        loss_item += loss.item()

        if curriculum_init:
            all_losses.extend(losses.tolist())

        if split == 'train' and not curriculum_init:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.scheduler:
                scheduler.step()

    return loss_item, len(dataloader), all_losses


def brad_do_epoch(model, split, dataloader, tokenizer, optimizer, args, scheduler=None, curriculum_init=False):
    loss_item = 0
    all_losses = []

    if split == 'train':
        model.train()
    elif split == 'test' or curriculum_init:
        model.eval()

    for batch in tqdm(dataloader):
        inputs_w, inputs_l, mask_w, mask_l, _, _ = full_tokenize(batch, tokenizer)
        losses = bradley_terry_loss(inputs_w, inputs_l, mask_w=mask_w, mask_l=mask_l, model=model)

        loss = losses.mean()
        loss_item += loss.item()
        #accurate_rankings = [l < np.log(2) for l in losses]


        if curriculum_init:
            all_losses.extend(losses.tolist())

        if split == 'train' and not curriculum_init:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.scheduler:
                scheduler.step()

    return loss_item, len(dataloader), all_losses

def brad_accuracy(model, dataloader, tokenizer):
    accuracy_item = 0

    model.eval()

    for batch in tqdm(dataloader):
        inputs_w, inputs_l, mask_w, mask_l, _, _ = full_tokenize(batch, tokenizer)
        with torch.no_grad():
            r_w = model(input_ids=inputs_w.to(DEVICE), attn_mask=mask_w.to(DEVICE)).squeeze()
            r_l = model(input_ids=inputs_l.to(DEVICE), attn_mask=mask_l.to(DEVICE)).squeeze()
            accuracy = (r_w > r_l).float().mean()

        accuracy_item += accuracy


    return accuracy_item, len(dataloader)
