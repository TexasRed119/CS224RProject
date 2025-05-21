import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from dpo.py import DPO_Preprocessor
import torch.nn.functional as F
from datasets import load_dataset
import argparse
import torch.optim as optim
from tqdm import tqdm

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
BRADLEY_TERRY_DATASET = "HuggingFaceH4/ultrafeedback_binarized"
device = 'cuda'

class RewardModel(nn.Module):
    def __init__(self):
        super(RewardModel, self).__init__()

        self.base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
        hidden_dim = self.base_model.config.hidden_size
        self.output_layer = nn.Linear(hidden_dim, 1)

        parameters = list(self.base_model.parameters()) + list(self.output_layer.parameters())
        self.optimizer = torch.optim.AdamW(parameters)

    def forward(self, input_ids, attn_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        return self.output_layer(last_hidden_state)

# gonna calculate my bradley terry loss
# bradley terry low-key sounds like a goated wide reciever from the 1990s
# if you told me he was on those troy aikman cowboy teams I would've believed you
def bradley_terry_loss(inputs_w, inputs_l, mask_w, mask_l, model):

    reward_w = model(input_ids=inputs_w, attention_mask=mask_w)
    reward_l = model(input_ids=inputs_l, attention_mask=mask_l)

    loss = F.logsigmoid(reward_w - reward_l).mean()

    return -loss

# bradley terry 
def main(args):

    model = RewardModel()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = load_dataset(BRADLEY_TERRY_DATASET, split="train_prefs")

    preprocessor = DPO_Preprocessor(tokenizer)
    train_dataset.map(preprocessor)

    # training
    for epoch in range(args.num_epochs):
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        for batch in tqdm(train_dataloader):
            inputs_w, inputs_l = batch["input_preferred"], batch["input_dispreferred"]
            mask_w, mask_l = batch["attention_mask_preferred"], batch["attention_mask_dispreferred"]
            loss = bradley_terry_loss(inputs_w, inputs_l, mask_w=mask_w, mask_l=mask_l, model=model)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-4)
    args = parser.parse_args()
    main(args)
