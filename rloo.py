import torch
import argparse
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
from bradley_terry import RewardModel
from bradley_terry import set_seed
import time
import json
import numpy as np
from curriculum_dataset import CurriculumDataset
from do_epoch import rloo_do_epoch, sft_do_epoch
from vllm import LLM, SamplingParams

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DPO_PATH = "models/dpo/epochs_4-batch_2-lr_1e-07-beta_0.1-seed_42-scheduler_False.pt"
BRADLEY_PATH = "models/brad/epochs_10-optim_adam-batch_4-lr_1e-05-seed_42-scheduler_False.pt"
COUNTDOWN_DATASET = "Jiayi-Pan/Countdown-Tasks-3to4"
SFT_DATASET = "Asap7772/cog_behav_all_strategies"
#RLOO_DATASET = "dpo_train.json"
#RLOO_VAL = "dpo_val.json"
DEVICE = 'cuda'
print(f"Using device: {DEVICE}")

# making a collate function because we are getting a bug: nums is not always the same length so its not wanting to get batched into tensors
# so instead we batch into dict of lists, so can be different lengths
def countdown_collate_fn(batch):
    batch_dict = {}
    for key in batch[0]:
        batch_dict[key] = [d[key] for d in batch]
    return batch_dict

def main(args):
    start_time = time.time()

    set_seed(args.seed)

    # load model from DPO (TODO: load SFT instead?)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, sliding_window=None).to(DEVICE)
    state_dict = torch.load(DPO_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    # load trained bradey terry reward model
    '''
    reward_model = RewardModel().to(DEVICE)
    brad_state = torch.load(BRADLEY_PATH, map_location=DEVICE)
    reward_model.load_state_dict(brad_state, strict=True)
    reward_model.eval()
    '''

    # vllm loading
    model.save_pretrained("./rloo_model")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained("./rloo_model")

    llm = LLM(model="./rloo_model")

    # sampling params for vllm
    sampling_params = SamplingParams(
        temperature=args.temp,
        top_p=0.9,
        max_tokens=1024,
        stop=None,
        n=args.k
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = None
    if args.scheduler:
        train_dataset = load_dataset(COUNTDOWN_DATASET, split="train[-1000:]")
        num_steps = 0
        for i in range(args.num_epochs):
            examples_in_epoch = ((i+1) / args.num_epochs) * len(train_dataset)
            if args.repeat_epochs is not None and str(i) in args.repeat_epochs.keys():
                examples_in_epoch *= int(args.repeat_epochs[str(i)])
            num_steps += int(examples_in_epoch / args.batch_size)
        print(f"Num training steps: {num_steps}")
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_steps)

    
    test_dataset = load_dataset(COUNTDOWN_DATASET, split="train[-2400:-2000]")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=countdown_collate_fn)

    sft_test_dataset = load_dataset(SFT_DATASET, split='test')
    sft_test_dataloader = torch.utils.data.DataLoader(sft_test_dataset, batch_size=args.batch_size)

    prev_indices = np.array([], dtype=int)
    prev_losses = None
    best_val_loss = float('inf')
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
                        dataset_name=COUNTDOWN_DATASET,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        args=args,
                        do_epoch=rloo_do_epoch,
                        cur_epoch=epoch,
                        num_epochs=args.num_epochs,
                        anti=anti,
                        prev_indices=prev_indices,
                        prev_losses=prev_losses
                    )
                    if args.static_curr and epoch == 0:  # if we don't want to recalculate losses every epoch
                        prev_losses = train_dataset.unseen_losses
                    prev_indices = np.copy(train_dataset.indices_to_train)
                    assert len(np.unique(prev_indices)) == len(prev_indices), "No repeated indexes in curriculum epoch."
                else:
                    train_dataset = load_dataset(COUNTDOWN_DATASET, split="train[-2000:]")
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=countdown_collate_fn)
            train_loss, num_batches, _ = rloo_do_epoch(model, llm, sampling_params,'train', train_dataloader, tokenizer, optimizer, args, scheduler=scheduler)
            print(f"Epoch: {epoch}, Train loss: {train_loss / num_batches}\n")
            with torch.no_grad():
                val_loss, num_batches, _ = rloo_do_epoch(model, llm, sampling_params, 'test', test_dataloader, tokenizer, optimizer, args, scheduler=None)
            print(f"Epoch: {epoch}, Val loss: {val_loss / num_batches}")

            with torch.no_grad():
                sft_val_loss, num_batches, _ = sft_do_epoch(model, 'test', sft_test_dataloader, tokenizer, optimizer, args, scheduler=None)
            print(f"Epoch: {epoch}, SFT Val loss: {sft_val_loss / num_batches}")
            
            if val_loss < best_val_loss:
                model_path = f'./models/rloo/epochs_{args.num_epochs}-k_{args.k}-temp_{args.temp}-batch_{args.batch_size}-lr_{args.lr}-seed_{args.seed}-scheduler_{args.scheduler}.pt'
                print(f'Saving best model: {model_path}')
                torch.save(
                    model.state_dict(),
                    model_path
                )
                best_val_loss = val_loss

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Wall time in minutes: {total_time / 60}")

    '''
    for epoch in range(args.num_epochs):
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
        for example in tqdm(train_dataloader):
            model.eval()
            batch_prompt = [example['prompt'][0] for i in range(args.batch_size)]
            tokenized_prompt = tokenizer(batch_prompt, return_tensors='pt', padding=True)
            x_and_y = model.generate(**tokenized_prompt, max_new_tokens=1)
            x_and_y = tokenizer.batch_decode(x_and_y)
            x_and_y = tokenizer(x_and_y)
            model.train()
            with torch.no_grad():
                rewards = reward_model(x_and_y['input_ids'], x_and_y['attention_mask'])
            
            # have no fear...tomas is here
            # for loop would be sooooo easy but matt's gonna give me shit
            # for calculating the baseline average across batch instead:
            # lets make a matrix with diagonal masked out...simple matrix multiplcation w rewwards will give us the leaving one out sums . then we just divide k - 1
            # current shape of rewards is gonna be (batch,1) multiple by our zero diagonal matrix of batch x batch
            diagonal_zero_matrix = torch.ones((args.batch_size, args.batch_size)) - torch.eye(args.batch_size)
            leave_one_out_sum = diagonal_zero_matrix @ rewards
            baselines = leave_one_out_sum / (args.batch_size - 1)
            # ^^ you might have to squeeze / unsqueeze depending on if rewards is (batch,) or (batch,1), but I think this will work as is

            # we gotta make a prompt mask again...gonna retokinze prompts without the padding
            prompt_unpadded = tokenizer(batch_prompt, return_tensors=None, padding=False)["input_ids"]
            prompt_lens = [len(p) for p in prompt_unpadded] 
            prompt_mask = torch.ones_like(x_and_y["input_ids"])
            for i in range(len(prompt_lens)):
                prompt_len = prompt_lens[i]
                prompt_mask[i, :prompt_len] = 0

            log_prob = compute_log_prob(model, x_and_y["input_ids"], x_and_y["attention_mask"], prompt_mask)

            loss = -((rewards - baselines) * log_prob).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    '''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('--k', type=int, default=4, help='number of samples per prompt')
    parser.add_argument('--temp', type=float, default=0.70, help='temperature in lmm sampling')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--scheduler', action='store_true')
    parser.add_argument('--curr_type', type=str, default='none')  # options: 'none', 'curriculum', 'anti'
    parser.add_argument('--static_curr', action='store_true', help='Changes type of curriculum learning')
    parser.add_argument('--repeat_epochs', type=json.loads, help="Specify dict of epochs to repeat, and how many times to repeat.")
    args = parser.parse_args()
    main(args)