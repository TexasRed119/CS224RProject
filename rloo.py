import torch
import argparse
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
from tqdm import tqdm
from bradley_terry import RewardModel
from dpo import compute_log_prob

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
            prompt_lens = [len(p) for p in prompt_tokens] 
            prompt_mask = torch.ones_like(inputs_preferred["input_ids"])
            for i in range(len(prompt_lens)):
                prompt_len = prompt_lens[i]
                prompt_mask[i, :prompt_len] = 0

            log_prob = compute_log_prob(model, x_and_y["input_ids"], x_and_y["attention_mask"], prompt_mask)

            loss = -((rewards - baselines) * log_prob).mean()
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