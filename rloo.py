import torch
import argparse
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
from tqdm import tqdm
from bradley_terry import RewardModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
RLOO_DATASET = "HuggingFaceH4/ultrafeedback_binarized"
device = 'cpu'

def main(args):
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    reward_model = RewardModel()  # todo: load trained reward model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = load_dataset(RLOO_DATASET, split="train_prefs")

    for epoch in range(args.num_epochs):
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
        for example in tqdm(train_dataloader):
            model.eval()
            batch_prompt = [example['prompt'][0] for i in range(args.batch_size)]
            tokenized_prompt = tokenizer(batch_prompt, return_tensors='pt', padding=True)
            x_and_y = model.generate(**tokenized_prompt, max_new_tokens=1)  # todo: change max_new_tokens
            model.train()
            with torch.no_grad():
                rewards = reward_model(x_and_y['input_ids'], x_and_y['attention_mask'])
            print()
            

            loss = 0
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=0.1)
    args = parser.parse_args()
    main(args)