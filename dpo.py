import torch
import torch.nn.functional as F
import argparse
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DPO_DATASET = "HuggingFaceH4/ultrafeedback_binarized"


class DPO_Preprocessor():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, element):
        prompt = element["prompt"]
        preferred = self.format(element["chosen"])
        dispreferred = self.format(element["rejected"])

        # have to use padding="max_length" and not padding=True since we are doing per element, not per batch
        input_preferred = self.tokenizer(prompt + preferred, padding="max_length", return_tensors="pt")
        input_dispreferred = self.tokenizer(prompt + dispreferred, padding="max_length", return_tensors="pt")

        # make mask for when we need to compute the loss without the prompt
        # this will be mulyiplied by the label log probs when computing the loss
        # first, compute prompt length so we can make prompt_mask
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        prompt_len = prompt_tokens.shape[1]
        # mask everything that is part of the prompt: 1s after prompt, 0s in prompt
        prompt_mask = torch.ones_like(input_preferred["input_ids"])
        prompt_mask[:, :prompt_len] = 0

        return {
            "input_preferred": input_preferred["input_ids"].squeeze(0),
            "input_dispreferred": input_dispreferred["input_ids"].squeeze(0),
            "attention_mask_preferred": input_preferred["attention_mask"].squeeze(0),
            "attention_mask_dispreferred": input_preferred["attention_mask"].squeeze(0),
            "prompt_mask": prompt_mask.squeeze(0)
            }
    
    def format(self, x):
        return x[1]["content"]


# todo: figure this out later, make it so we can use vanilla or cumulative
def load_data(cumulative=False):
    pass


# function to calcualte the log_probs and then use the prompt mask to mask the prompt before calcualting loss
# we calculate all 4 log probs in equation using this
def compute_log_prob(model, inputs, attention_mask, prompt_mask):

    outputs = model(input_ids=inputs, attention_mask=attention_mask)
    logits = outputs.logits 
    pred_probs = F.softmax(logits, dim=-1)

    # model is predicting next token, so we need to shift everything by one to compare to ground truth (which is the input in our case)
    # right now we have shape of (batch, sequence, vocab), so adjust over sequence dimension
    pred_probs = pred_probs[:, :-1, :]  # don't need last token prediction
    target_ids = inputs[:, 1:]  # our ground truth labels
    prompt_mask = prompt_mask[:, 1:]

    # log-probabilities of the target tokens
    target_preds = torch.gather(pred_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)

    log_probs = torch.log(target_preds)

    # we don't care about the prediction loss for the prompts, only the responses
    log_probs = log_probs * prompt_mask 

    return log_probs.sum(dim=-1)  # need to sum over all of the tokens in the response

# apparently common values for beta are 0.1, 0.5, possible range of 0.01 -> 10. High values will overfit, low values will underfit
# 0.1 beta with 10 epochs might be similar to 1 epoch with 10 beta
def dpo_loss(inputs_w, inputs_l, mask_w, mask_l, model, ref_model, beta, prompt_mask):

    logprob_model_w = compute_log_prob(model, inputs_w, mask_w, prompt_mask)
    logprob_model_l = compute_log_prob(model, inputs_l, mask_l, prompt_mask)

    # reference model is frozen, so don't calculate gradients
    with torch.no_grad():
        logprob_ref_w = compute_log_prob(ref_model, inputs_w, mask_w, prompt_mask)
        logprob_ref_l = compute_log_prob(ref_model, inputs_l, mask_l, prompt_mask)

    # changed the fomula a little, but mathematically the same. was easier to take 4 log_probs and then subtract, instead doing division and then log
    logits = beta * ((logprob_model_w - logprob_ref_w) - (logprob_model_l -  logprob_ref_l))
    loss = -F.logsigmoid(logits).mean()
    return loss


def main(args):

    # model and ref_model these will both be the model from sft, thank you MATTHEUS!
    # ref_model is the frozen policy
    # AYO! YOU FROM BROOKLYN??
    # torch load will be used?
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    ref_model.eval()  # freeze this bad boy like frozone
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = load_dataset(DPO_DATASET, split="train_prefs")

    # using .map to get prompt + response inputs...fuck you mattheus I aint no bum
    preprocessor = DPO_Preprocessor(tokenizer)
    train_dataset.map(preprocessor)

    # training
    for epoch in range(args.num_epochs):
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        for batch in tqdm(train_dataloader):
            inputs_w, inputs_l = batch["input_preferred"], batch["input_dispreferred"]
            mask_w, mask_l = batch["attention_mask_preferred"], batch["attention_mask_dispreferred"]
            prompt_mask = batch["prompt_mask"]
            loss = dpo_loss(inputs_w, inputs_l, mask_w=mask_w, mask_l=mask_l, model=model, ref_model=ref_model, beta=args.beta, prompt_mask=prompt_mask)
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