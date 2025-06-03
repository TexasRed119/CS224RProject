from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
import argparse
from datasets import load_dataset
import torch
import torch.optim as optim
import random
import numpy as np
from do_epoch import sft_do_epoch
from curriculum_dataset import CurriculumDataset
import time
import json

SFT_DATASET = "Asap7772/cog_behav_all_strategies"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DPO_PATH = "models/dpo/epochs_3-batch_6-lr_1e-05-beta_0.01-seed_42.pt"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args):
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", sliding_window=None).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    state_dict = torch.load(DPO_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    test_dataset = load_dataset(SFT_DATASET, split='test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

    with torch.no_grad():
        val_loss, num_batches, _ = sft_do_epoch(model, 'test', test_dataloader, tokenizer, optimizer, args, scheduler=None)
    print(f"Val loss: {val_loss / num_batches}\n")

# args dont mean shit here
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-6)
    parser.add_argument('--scheduler', action='store_true')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--curr_type', type=str, default='curriculum')  # options: 'none', 'curriculum', 'anti'
    parser.add_argument('--static_curr', action='store_true', help='Changes type of curriculum learning')
    parser.add_argument('--repeat_epochs', type=json.loads, help="Specify dict of epochs to repeat, and how many times to repeat.")
    """
    With repeat_epochs, to repeat epoch 0 twice run: --repeat_epochs '{"0": "2"}'
    """
    args = parser.parse_args()
    main(args)
