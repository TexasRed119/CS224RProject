from torch.utils.data import Dataset
from datasets import load_dataset

SFT_DATASET = "Asap7772/cog_behav_all_strategies"

# THIS FILE IS UNUSED
# todo: pre-process all the tokenization that is in sft.py

class WarmStartDataset(Dataset):

    def __init__(self, tokenizer, split):
        self.dataset = load_dataset(SFT_DATASET, split=split)
        # self.completion_ids = tokenizer(
        #     self.dataset['completion'],
        #     return_attention_mask=False,
        #     padding=True,
        #     return_tensors='pt'
        # )['input_ids']
        # query_ids = tokenizer(
        #     self.dataset['query'],
        #     return_attention_mask=False,
        #     padding
        #     return_tensors='pt'
        # )['input_ids']
        print()


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        return {'query_ids': self.query_ids[idx], 'completion_ids': self.completion_ids[idx]}
