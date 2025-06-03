import torch
import torch.nn.functional as F
import argparse
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2ForCausalLM, Qwen2Config, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
import time
import random
import numpy as np
import json
from do_epoch import dpo_do_epoch
# from the_streets import a_couple_of_gs
from sft import SFT_DATASET

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
COUNTDOWN_DATASET = "Jiayi-Pan/Countdown-Tasks-3to4"
DPO_DATASET = "dpo_dataset.json"
#DPO_DATASET = "dpo_dataset_small.json"
SFT_PATH = "BEST_epochs_6-batch_4-lr_1e-05-seed_42-curr_type_none-scheduler_True-static_False-repeat_epochs_None.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# we needed this when we were making circles...now its squares
def format_hug(x):
        return x[1]["content"]

# function to calcualte the log_probs and then use the prompt mask to mask the prompt before calcualting loss
# we calculate all 4 log probs in equation using this
def compute_log_prob(model, inputs, attention_mask, prompt_mask):
    outputs = model(input_ids=inputs.to(device), attention_mask=attention_mask.to(device))
    logits = outputs.logits
    print("DPO logits stats: ", logits.mean().item(), logits.std().item())

    pred_probs = F.softmax(logits, dim=-1)

    # model is predicting next token, so we need to shift everything by one to compare to ground truth (which is the input in our case)
    # right now we have shape of (batch, sequence, vocab), so adjust over sequence dimension
    pred_probs = pred_probs[:, :-1, :]  # don't need last token prediction
    target_ids = inputs[:, 1:]  # our ground truth labels
    prompt_mask = prompt_mask[:, 1:]

    # log-probabilities of the target tokens
    target_preds = torch.gather(pred_probs.to(device), 2, target_ids.unsqueeze(-1).to(device)).squeeze(-1)

    log_probs = torch.log(target_preds)

    # we don't care about the prediction loss for the prompts, only the responses
    log_probs = log_probs * prompt_mask.to(device)

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
    loss = -F.logsigmoid(logits)
    return loss

def full_tokenize(batch, tokenizer):
    inputs_preferred = []
    inputs_dispreferred = []
    # getting prompts so we can make prompt mask
    prompts = []
    for i in range(len(batch['prompt'])):
        prompts.append(batch['prompt'][i])
        # Add space between prompt and response
        inputs_preferred.append(batch['prompt'][i] + " " + batch['chosen'][i])
        inputs_dispreferred.append(batch['prompt'][i] + " " + batch['rejected'][i])

    # preferred
    inputs_preferred = tokenizer(
        inputs_preferred,
        padding=True,
        return_tensors='pt'
    )
    inputs_w = inputs_preferred["input_ids"]
    mask_w = inputs_preferred["attention_mask"]

    # dispreffered
    inputs_dispreferred = tokenizer(
        inputs_dispreferred,
        padding=True,
        return_tensors='pt'
    )
    inputs_l = inputs_dispreferred["input_ids"]
    mask_l = inputs_dispreferred["attention_mask"]

    
    # making the masks, can't return a tensor because then they will be different lengths
    # need to use 2 different masks due to different sequence lengths
    prompt_tokens = tokenizer(
        prompts,
        padding=False,
        return_tensors=None
    )["input_ids"]
    # mask everything that is part of the prompt: 1s after prompt, 0s in prompt
    prompt_lens = [len(p) for p in prompt_tokens] 
    
    #prompt_tokenized = tokenizer(prompts, padding=True, return_tensors='pt')
    #prompt_lens = (prompt_tokenized['attention_mask']).sum(dim=1)

    prompt_mask_w = torch.ones_like(inputs_preferred["input_ids"])
    prompt_mask_l = torch.ones_like(inputs_dispreferred["input_ids"])
    for i in range(len(prompt_lens)):
        prompt_len = prompt_lens[i]
        prompt_mask_w[i, :prompt_len] = 0
        prompt_mask_l[i, :prompt_len] = 0

    return inputs_w, inputs_l, mask_w, mask_l, prompt_mask_w, prompt_mask_l

def main(args):
    start_time = time.time()

    set_seed(args.seed)

    # model and ref_model these will both be the model from sft, thank you MATTHEUS!
    # ref_model is the frozen policy
    # AYO! YOU FROM BROOKLYN??
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, sliding_window=None)
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, sliding_window=None)
    ref_model.eval()  # freeze this bad boy like frozone

    # Load model with appropriate device mapping
    # COMMMENTED OUT FOR DEBUGGING
    state_dict = torch.load(SFT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    ref_model.load_state_dict(state_dict)

    # Move models to device
    model = model.to(device)
    ref_model = ref_model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


    if args.scheduler:
        # Load dataset i made
        dataset_dict = load_dataset("json", data_files=DPO_DATASET)
        train_dataset = dataset_dict["train"]
        num_steps = 0
        for i in range(args.num_epochs):
            examples_in_epoch = ((i+1) / args.num_epochs) * len(train_dataset)
            if args.repeat_epochs is not None and str(i) in args.repeat_epochs.keys():
                examples_in_epoch *= int(args.repeat_epochs[str(i)])
            num_steps += int(examples_in_epoch / args.batch_size)
        print(f"Num training steps: {num_steps}")
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_steps)

    test_dataset = load_dataset(SFT_DATASET, split='test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

    prev_indices = np.array([])
    prev_losses = None
    for epoch in range(args.num_epochs):
        if args.repeat_epochs is not None and str(epoch) in args.repeat_epochs.keys():
            times_to_repeat = int(args.repeat_epochs[str(epoch)])
        else:
            times_to_repeat = 1
        for i in range(times_to_repeat):
            if i == 0:  # only create train_dataset if on first repeat
                if args.curr_type in ['curriculum', 'anti']:
                    anti = args.curr_type == 'anti'
                    train_dataset = CurriculumDataset(
                        model=model,
                        split='train',
                        dataset_name=DPO_DATASET,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        args=args,
                        do_epoch=dpo_do_epoch,
                        cur_epoch=epoch,
                        num_epochs=args.num_epochs,
                        anti=anti,
                        prev_indices=prev_indices,
                        prev_losses=prev_losses,
                        is_json=True
                    )
                    if args.static_curr and epoch == 0:  # if we don't want to recalculate losses every epoch
                        prev_losses = train_dataset.unseen_losses
                    prev_indices = np.copy(train_dataset.indices_to_train)
                    assert len(np.unique(prev_indices)) == len(prev_indices), "No repeated indexes in curriculum epoch."
                else:
                    dataset_dict = load_dataset("json", data_files=DPO_DATASET)
                    train_dataset = dataset_dict["train"]
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            train_loss, num_batches, _ = dpo_do_epoch(model, ref_model, 'train', train_dataloader, tokenizer, optimizer, args, scheduler=scheduler)
            print(f"Epoch: {epoch}, Train loss: {train_loss / num_batches}\n")
            with torch.no_grad():
                val_loss, num_batches, _ = dpo_do_epoch(model, ref_model, 'test', test_dataloader, tokenizer, optimizer, args, scheduler=None)
            print(f"Epoch: {epoch}, Val loss: {val_loss / num_batches}\n")

    '''
    # training
    for epoch in range(args.num_epochs):
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        for batch in tqdm(train_dataloader):

            inputs_w, inputs_l, mask_w, mask_l, prompt_mask_w, prompt_mask_l = full_tokenize(batch, tokenizer)

            loss = dpo_loss(inputs_w, inputs_l, mask_w=mask_w, mask_l=mask_l, model=model, ref_model=ref_model, beta=args.beta, prompt_mask_w=prompt_mask_w, prompt_mask_l=prompt_mask_l)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
        
        print("This is the loss: ")
        print(loss.item())
    '''

    model_path = f'./dpo/epochs_{args.num_epochs}-batch_{args.batch_size}-lr_{args.lr}-beta_{args.beta}-seed_{args.seed}.pt'
    print(f'\n{model_path}\n')
    torch.save(
        model.state_dict(),
        model_path
    )

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Wall time in minutes: {total_time / 60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-10)
    parser.add_argument('--beta', type=float, default=1e-10)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--scheduler', action='store_true')
    parser.add_argument('--curr_type', type=str, default='curriculum')  # options: 'none', 'curriculum', 'anti'
    parser.add_argument('--static_curr', action='store_true', help='Changes type of curriculum learning')
    parser.add_argument('--repeat_epochs', type=json.loads, help="Specify dict of epochs to repeat, and how many times to repeat.")
    args = parser.parse_args()
    main(args)