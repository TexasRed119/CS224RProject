import torch
import argparse
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
import time
import random
import numpy as np
import json
from do_epoch import dpo_do_epoch, sft_do_epoch
from curriculum_dataset import CurriculumDataset
# from the_streets import a_couple_of_gs
from sft import SFT_DATASET

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
COUNTDOWN_DATASET = "Jiayi-Pan/Countdown-Tasks-3to4"
DPO_DATASET = "dpo_train.json"
DPO_VAL = "dpo_val.json"
SFT_PATH = "models/sft/BEST_epochs_6-batch_4-lr_1e-05-seed_42-curr_type_none-scheduler_True-static_False-repeat_epochs_None.pt"
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

    scheduler = None
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

    test_dict = load_dataset("json", data_files=DPO_VAL)
    test_dataset = test_dict["train"]  # ignore that this says train...its still test dataset
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

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
                        ref_model=ref_model
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
            print(f"Epoch: {epoch}, Val loss: {val_loss / num_batches}")

            if val_loss < best_val_loss:
                model_path = f'./models/dpo/epochs_{args.num_epochs}-batch_{args.batch_size}-lr_{args.lr}-beta_{args.beta}-seed_{args.seed}-scheduler_{args.scheduler}.pt'
                print(f'Saving best model: {model_path}')
                torch.save(
                    model.state_dict(),
                    model_path
                )
                best_val_loss = val_loss

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Wall time in minutes: {total_time / 60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-5)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--scheduler', action='store_true')
    parser.add_argument('--curr_type', type=str, default='curriculum')  # options: 'none', 'curriculum', 'anti'
    parser.add_argument('--static_curr', action='store_true', help='Changes type of curriculum learning')
    parser.add_argument('--repeat_epochs', type=json.loads, help="Specify dict of epochs to repeat, and how many times to repeat.")
    args = parser.parse_args()
    main(args)