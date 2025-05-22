import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_NEW_TOKENS = 256

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_completion(model, tokenizer, prompt, prompt_is_tokens=False, max_new_tokens=MAX_NEW_TOKENS):
    model.eval()
    
    if prompt_is_tokens:
        # tokenize the prompt
        inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(device)
    else:
        inputs = prompt_is_tokens
        
    
    prompt_length = inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,  
            top_p=0.9,      
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3
        )
    
    generated_tokens = outputs[0][prompt_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text.strip()

def main():
    # lay seed
    set_seed(42)
    
    print(f"Using device: {device}")
    
    # load model
    print("Loading model and tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    # load weights
    print("\nLoading trained weights")
    checkpoint_path = 'epochs_4-batch_4-lr_1e-06.pt'
    
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Weights loaded successfully!")
    
    except Exception as e:
        print(f"Error loading weights: {e}")

    model = model.to(device)
    model.eval()
    
    # Load evaluation dataset
    print("\nLoading dataset")
    eval_dataset = load_dataset("Asap7772/cog_behav_all_strategies", split="test")
    
    print("\nGenerating completions for samples:\n")
    for i in range(min(5, len(eval_dataset))):
        example = eval_dataset[i]
        
        print(f"\nSample {i+1}:")
        print(f"Query: {example['query']}")
        print(f"Ground truth: {example['completion']}")
        
        # generation
        print("Generating model completion")
        try:
            model_completion = generate_completion(model, tokenizer, example['query'])
            print(f"Model's completion: {model_completion}")
        except Exception as e:
            print(f"Error generating completion: {e}")
        
        print("-" * 80)

if __name__ == '__main__':
    main()