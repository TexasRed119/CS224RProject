import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2ForCausalLM, Qwen2Config
from datasets import load_dataset
from vllm import LLM
# rule-based reward function provided by gandhi
# GANDHI! HE'S BACK
# "I am not gandhi the grey...I am gandhi the white...and i come back to you now at the turn of the tide" 
from countdown_eval import compute_score
from countdown_eval import prompt_template



MODEL_NAME = "Qwen/Qwen2.5-0.5B"
COUNTDOWN_DATASET = "Jiayi-Pan/Countdown-Tasks-3to4"
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
    prompts = []
    chosen = []
    rejected = []

    # I was gonna loop through prompts, and then pass the entire list of prompts to lmm.generate..
    # but I would just have to loop through anyways to score and label responses...so I'm doing it like this
    for example in dataset:
        prompt = prompt_template(example["nums"], example["target"])
        prompts.append(prompt)

        
        while no_tie:
            output1, output2 = llm.generate([prompt, prompt])

            score1 = compute_score(output1, example)
            score2 = compute_score(output2, example)

            if score1 > score2: 
                chosen.append(output1)
                rejected.append(output2)
            elif
    
    return prompts

def main():
    
    set_seed(args.seed)

    # model will be the model from sft, thank you MATTHEUS!
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Load model with appropriate device mapping
    # COMMMENTED OUT FOR DEBUGGING
    #state_dict = torch.load('./models/sft/epochs_6-batch_4-lr_1e-06-seed_42.pt', map_location=device)
    #model.load_state_dict(state_dict)

    base_model.save_pretrained("./my_hf_model")
    tokenizer = AutoTokenizer.from_pretrained("base-model-name")
    tokenizer.save_pretrained("./my_hf_model")

    # use with vLLM
    llm = LLM(model="./my_hf_model")

    llm = llm.to(device)

    dataset = load_dataset(COUNTDOWN_DATASET, split="train")

    # make all the features we need for the new dataset (prompts, preferred, dispreferred)
    prompts, chosen, rejected = make_prompts(dataset)

    # save dataset to .json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    main(args)