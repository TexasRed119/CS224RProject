from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from datasets import load_dataset
import torch
import torch.optim as optim
import random
import numpy as np
from do_epoch import sft_do_epoch
from curriculum_dataset import CurriculumDataset
import time

SFT_DATASET = "Asap7772/cog_behav_all_strategies"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args):
    start_time = time.time()

    # Set random seed for reproducibility
    set_seed(args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", sliding_window=None).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    test_dataset = load_dataset(SFT_DATASET, split='test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

    prev_indices = np.array([])
    for epoch in range(args.num_epochs):
        if args.curr_type in ['curriculum', 'anti']:
            anti = args.curr_type == 'anti'
            train_dataset = CurriculumDataset(
                model=model,
                split='train',
                dataset_name=SFT_DATASET,
                tokenizer=tokenizer,
                optimizer=optimizer,
                args=args,
                do_epoch=sft_do_epoch,
                cur_epoch=epoch,
                num_epochs=args.num_epochs,
                anti=anti,
                prev_indices=prev_indices
            )
            prev_indices = np.copy(train_dataset.indices_to_train)
            assert len(np.unique(prev_indices)) == len(prev_indices), "No repeated indexes in curriculum epoch."
        else:
            train_dataset = load_dataset(SFT_DATASET, split='train')
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        train_loss, num_batches, _ = sft_do_epoch(model, 'train', train_dataloader, tokenizer, optimizer, args)
        print(f"Epoch: {epoch}, Train loss: {train_loss / num_batches}\n")
        with torch.no_grad():
            val_loss, num_batches, _ = sft_do_epoch(model, 'test', test_dataloader, tokenizer, optimizer, args)
        print(f"Epoch: {epoch}, Val loss: {val_loss / num_batches}\n")

    model_path = f'./models/sft/epochs_{args.num_epochs}-batch_{args.batch_size}-lr_{args.lr}-seed_{args.seed}-curr_type_{args.curr_type}.pt'
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
    parser.add_argument('--num_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-6)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--curr_type', type=str, default='curriculum')  # options: 'none', 'curriculum', 'anti'
    args = parser.parse_args()
    main(args)
