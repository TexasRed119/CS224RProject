from torch.utils.data import Dataset
from datasets import load_dataset
from sft import do_epoch

class WarmStartDataset(Dataset):

    # cur_epoch should be zero_indexed
    def __init__(self, model, split, dataset_name, tokenizer, optimizer, args, do_epoch, is_curriculum, cur_epoch, num_epochs):
        self.dataset = load_dataset(dataset_name, split=split)
        if is_curriculum:
            _, _, all_losses = do_epoch(model, split, self.dataset, tokenizer, optimizer, args, is_curriculum)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        return {'query_ids': self.query_ids[idx], 'completion_ids': self.completion_ids[idx]}
