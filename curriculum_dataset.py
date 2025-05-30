from torch.utils.data import Dataset
from datasets import load_dataset
import torch

class CurriculumDataset(Dataset):

    # cur_epoch should be zero_indexed
    def __init__(
            self,
            model,
            split,
            dataset_name,
            tokenizer,
            optimizer,
            args,
            do_epoch,
            cur_epoch,
            num_epochs
        ):

        dataset = load_dataset(dataset_name, split=split)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        _, _, all_losses = do_epoch(model, split, dataloader, tokenizer, optimizer, args, True)
        print()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {'query_ids': self.query_ids[idx], 'completion_ids': self.completion_ids[idx]}
