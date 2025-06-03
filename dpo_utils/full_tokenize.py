import torch

def full_tokenize(batch, tokenizer):
    inputs_preferred = []
    inputs_dispreferred = []
    # getting prompts so we can make prompt mask
    prompts = []
    for i in range(len(batch['prompt'])):
        prompts.append(batch['prompt'][i])
        # Add space between prompt and response
        inputs_preferred.append(batch['prompt'][i] + " " + batch['chosen'][i])
        inputs_dispreferred.append(batch['prompt'][i] + " " + batch['rejected'][i])

    # preferred
    inputs_preferred = tokenizer(
        inputs_preferred,
        padding=True,
        return_tensors='pt'
    )
    inputs_w = inputs_preferred["input_ids"]
    mask_w = inputs_preferred["attention_mask"]

    # dispreffered
    inputs_dispreferred = tokenizer(
        inputs_dispreferred,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    inputs_l = inputs_dispreferred["input_ids"]
    mask_l = inputs_dispreferred["attention_mask"]

    
    # making the masks, can't return a tensor because then they will be different lengths
    # need to use 2 different masks due to different sequence lengths
    prompt_tokens = tokenizer(
        prompts,
        padding=False,
        return_tensors=None
    )["input_ids"]
    # mask everything that is part of the prompt: 1s after prompt, 0s in prompt
    prompt_lens = [len(p) for p in prompt_tokens] 
    
    #prompt_tokenized = tokenizer(prompts, padding=True, return_tensors='pt')
    #prompt_lens = (prompt_tokenized['attention_mask']).sum(dim=1)

    prompt_mask_w = torch.ones_like(inputs_preferred["input_ids"])
    prompt_mask_l = torch.ones_like(inputs_dispreferred["input_ids"])
    for i in range(len(prompt_lens)):
        prompt_len = prompt_lens[i]
        prompt_mask_w[i, :prompt_len] = 0
        prompt_mask_l[i, :prompt_len] = 0

    return inputs_w, inputs_l, mask_w, mask_l, prompt_mask_w, prompt_mask_l