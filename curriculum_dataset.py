from torch.utils.data import Dataset as TorchDataset
from datasets import load_dataset, Dataset
import torch
import numpy as np

class CurriculumDataset(TorchDataset):

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

        dataset = load_dataset(dataset_name, split=split)  # TODO: change back to loading split
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        _, _, all_losses = do_epoch(model, split, dataloader, tokenizer, optimizer, args, True)
        
        percentage_of_dataset_to_keep = (cur_epoch + 1) / num_epochs
        index_cutoff = int(len(dataset) * percentage_of_dataset_to_keep)  # might have an off-by-one error but not that deep
         # all indexes with loss smaller than index_cutoff go at the start of the array
        chosen_indices = np.argpartition(all_losses, index_cutoff)[:index_cutoff]
        self.curriculum_dataset = Dataset.from_dict(dataset[chosen_indices])

    def __len__(self):
        return len(self.curriculum_dataset)

    def __getitem__(self, idx):
        return self.curriculum_dataset[idx]
