import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from dpo_utils.full_tokenize import full_tokenize
import torch.nn.functional as F
from datasets import load_dataset
import argparse
import torch.optim as optim
from tqdm import tqdm
import random
import numpy as np
import time
import json
from do_epoch import brad_do_epoch, sft_do_epoch, brad_accuracy
from curriculum_dataset import CurriculumDataset
# from the_streets import a_couple_of_gs
from sft import SFT_DATASET


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
COUNTDOWN_DATASET = "Jiayi-Pan/Countdown-Tasks-3to4"
DPO_PATH = "models/dpo/epochs_4-batch_2-lr_1e-07-beta_0.1-seed_42-scheduler_False.pt"
DPO_DATASET = "dpo_train.json"
DPO_VAL = "dpo_val.json"
DEVICE = 'cuda'
print(f"Using device: {DEVICE}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class RewardModel(nn.Module):
    def __init__(self):
        super(RewardModel, self).__init__()

        self.base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", sliding_window=None)
        hidden_dim = self.base_model.config.hidden_size
        self.output_layer = nn.Linear(hidden_dim, 1)  # todo: initialize layer
        ''' for if we want to manually init layer
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_layer.bias)
        '''

        parameters = list(self.base_model.parameters()) + list(self.output_layer.parameters())
        #self.optimizer = torch.optim.AdamW(parameters)

    def forward(self, input_ids, attn_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        return self.output_layer(last_hidden_state)

# bradley terry 
def main(args):
    start_time = time.time()

    set_seed(args.seed)

    model = RewardModel().to(DEVICE)
    if args.optimizer == "adam":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    if args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    scheduler = None
    if args.scheduler:
        # Load dataset i made
        dataset_dict = load_dataset("json", data_files=DPO_DATASET)
        train_dataset = dataset_dict["train"]
        num_steps = 0
        if args.curr_type == 'none':  # todo: check if curr_type none for scheduler in sft.
            num_steps = int(args.num_epochs * len(train_dataset) / args.batch_size)
        else:
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
                        do_epoch=brad_do_epoch,
                        cur_epoch=epoch,
                        num_epochs=args.num_epochs,
                        anti=anti,
                        prev_indices=prev_indices,
                        prev_losses=prev_losses,
                        is_brad=True
                    )
                    if args.static_curr and epoch == 0:  # if we don't want to recalculate losses every epoch
                        prev_losses = train_dataset.unseen_losses
                    prev_indices = np.copy(train_dataset.indices_to_train)
                    assert len(np.unique(prev_indices)) == len(prev_indices), "No repeated indexes in curriculum epoch."
                else:
                    dataset_dict = load_dataset("json", data_files=DPO_DATASET)
                    train_dataset = dataset_dict["train"]
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            train_loss, num_batches, _ = brad_do_epoch(model, 'train', train_dataloader, tokenizer, optimizer, args, scheduler=scheduler)
            print(f"Epoch: {epoch}, Train loss: {train_loss / num_batches}\n")
            with torch.no_grad():
                val_loss, num_batches, _ = brad_do_epoch(model, 'test', test_dataloader, tokenizer, optimizer, args, scheduler=None)
            
            avg_val_loss = val_loss / num_batches
            print(f"Epoch: {epoch}, Val loss: {avg_val_loss}")

            # instead of sft loss here (since this is a reward model)... 
            # I should instead measure how often it is correct (preferred > dispreferred)? tho that is kinda what the loss function is doing
            # i implemented the accuracy...but it takes a little while so I'll skip for now
            '''
            with torch.no_grad():
                train_accuracy, num_batches = brad_accuracy(model, train_dataloader, tokenizer)
                avg_train_accuracy = train_accuracy / num_batches
                val_accuracy, num_batches = brad_accuracy(model, test_dataloader, tokenizer)
                avg_val_accuracy = val_accuracy / num_batches

            print(f"Epoch: {epoch}, Train accuracy: {avg_train_accuracy}")
            print(f"Epoch: {epoch}, Val accuracy: {avg_val_accuracy}")
            '''
            if avg_val_loss < best_val_loss:
                model_path = f'./models/brad/epochs_{args.num_epochs}-optim_{args.optimizer}-batch_{args.batch_size}-lr_{args.lr}-seed_{args.seed}-scheduler_{args.scheduler}.pt'
                print(f'Saving best model: {model_path}')
                torch.save(
                    model.state_dict(),
                    model_path
                )
                best_val_loss = avg_val_loss
    
    '''
    # training
    for epoch in range(args.num_epochs):
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        for batch in tqdm(train_dataloader):
            inputs_w, inputs_l, mask_w, mask_l, _, _ = full_tokenize(batch)
            loss = bradley_terry_loss(inputs_w, inputs_l, mask_w=mask_w, mask_l=mask_l, model=model)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if val_loss < best_val_loss:
                model_path = f'./models/dpo/epochs_{args.num_epochs}-batch_{args.batch_size}-lr_{args.lr}-beta_{args.beta}-seed_{args.seed}-scheduler_{args.scheduler}.pt'
                print(f'Saving best model: {model_path}')
                torch.save(
                    model.state_dict(),
                    model_path
                )
                best_val_loss = val_loss
    '''

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Wall time in minutes: {total_time / 60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='adam')  # options: 'adam', 'rmsprop' 
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--scheduler', action='store_true')
    parser.add_argument('--curr_type', type=str, default='none')  # options: 'none', 'curriculum', 'anti'
    parser.add_argument('--static_curr', action='store_true', help='Changes type of curriculum learning')
    parser.add_argument('--repeat_epochs', type=json.loads, help="Specify dict of epochs to repeat, and how many times to repeat.")
    args = parser.parse_args()
    main(args)
