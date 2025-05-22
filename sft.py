from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from datasets import load_dataset
import torch
import torch.optim as optim
import datetime
from tqdm import tqdm
import random
import numpy as np
from eval import generate_completion

SFT_DATASET = "Asap7772/cog_behav_all_strategies"
device = 'cuda'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def do_epoch(model, split, dataset, tokenizer, optimizer, args):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    loss_item = 0

    if split == 'train':
        model.train()
    elif split == 'test':
        model.eval()

    for batch in tqdm(dataloader):
        query_and_completion = []
        for i in range(len(batch['query'])):
            query_and_completion.append(batch['query'][i] + batch['completion'][i])
        query_and_completion = tokenizer(
            query_and_completion,
            padding=True,
            return_tensors='pt'
        )

        output = model(
            query_and_completion['input_ids'].to(device),
            attention_mask=query_and_completion['attention_mask'].to(device)
        )

        loss = 0
        for i in range(len(batch['query'])):
            query_ids = tokenizer.encode(batch['query'][i])
            completion_ids = tokenizer.encode(batch['completion'][i], return_tensors='pt')
            pred_start = len(query_ids)-1
            num_preds = len(completion_ids[0])
            pred_logits = output['logits'][i][pred_start:pred_start+num_preds]
            pred_probs = torch.nn.functional.softmax(pred_logits, dim=1)
            completion_ids_col = completion_ids.reshape((-1, 1)).to(device)

            target_preds = torch.gather(pred_probs, 1, completion_ids_col)
            target_preds = torch.log(target_preds)
            x_loss = -torch.sum(target_preds)
            loss += x_loss
        loss = loss / len(batch['query'])
        loss_item += loss.item()
        
        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    example_generation = generate_completion(
        model,
        tokenizer,
        query_and_completion.to(device),
        prompt_is_tokens=True,
        max_new_tokens=None
    )
    print("Example Generation: ")
    print(example_generation)
    model.train()

    return loss_item, len(dataloader)

def main(args):
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    train_dataset = load_dataset(SFT_DATASET, split='train')
    test_dataset = load_dataset(SFT_DATASET, split='test')

    for epoch in range(args.num_epochs):
        train_loss, num_batches = do_epoch(model, 'train', train_dataset, tokenizer, optimizer, args)
        print(f"Epoch: {epoch}, Train loss: {train_loss / num_batches}")
        print("")
        val_loss, num_batches = do_epoch(model, 'test', test_dataset, tokenizer, optimizer, args)
        print(f"Epoch: {epoch}, Val loss: {val_loss / num_batches}")

    torch.save(
        model.state_dict(),
        f'./models/sft/epochs_{args.num_epochs}-batch_{args.batch_size}-lr_{args.lr}-seed_{args.seed}.pt'
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-6)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    main(args)
