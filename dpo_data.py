import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2ForCausalLM, Qwen2Config
from datasets import load_dataset
from vllm import LLM
import json
# rule-based reward function provided by gandhi
# GANDHI! HE'S BACK
# "I am not gandhi the grey...I am gandhi the white...and i come back to you now at the turn of the tide" 
from countdown_eval import compute_score
from countdown_eval import prompt_template



MODEL_NAME = "Qwen/Qwen2.5-0.5B"
SFT_PATH = "BEST_epochs_6-batch_4-lr_1e-05-seed_42-curr_type_none-scheduler_True-static_False-repeat_epochs_None.pt"
COUNTDOWN_DATASET = "Jiayi-Pan/Countdown-Tasks-3to4"
DPO_PATH = "dpo_dataset.json" 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# make prompts using prompt template
# make 2 responses for each prompt, compute score and then label preferred or dispreferred
# note: if tie, just delete one and have model generate another response that will be better or worse
def make_features(llm, dataset):
    dpo_dataset = []

    # I was gonna loop through prompts, and then pass the entire list of prompts to lmm.generate..
    # but I would just have to loop through anyways to score and label responses...so I'm doing it like this
    for example in dataset:
        prompt = prompt_template(example["nums"], example["target"])
        chosen = None
        rejected = None

        # while tie stil exists, break otherwise
        while True:
            output1, output2 = llm.generate([prompt, prompt])

            score1 = compute_score(output1, example)
            score2 = compute_score(output2, example)

            if score1 > score2: 
                chosen = output1
                rejected = output2
                break
            elif score2 > score1:
                chosen = output2
                rejected = output1
                break
        
        dpo_dataset.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })
    
    return dpo_dataset

def main():
    
    set_seed(args.seed)

    # model will be the model from sft, thank you MATTHEUS!
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Load model with appropriate device mapping
    # COMMMENTED OUT FOR DEBUGGING
    state_dict = torch.load(SFT_PATH, map_location=device)
    model.load_state_dict(state_dict)

    base_model.save_pretrained(SFT_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(SFT_PATH)

    # use with vLLM
    llm = LLM(model="./my_hf_model")

    llm = llm.to(device)

    dataset = load_dataset(COUNTDOWN_DATASET, split="train")

    # make all the features we need for the new dataset (prompts, preferred, dispreferred)
    dpo_dataset = make_features(dataset)

    # save dataset to .json
    with open(DPO_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    main(args)