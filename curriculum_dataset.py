from torch.utils.data import Dataset as TorchDataset, Subset
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
            num_epochs,
            anti,  # TODO: implement curriculum learning
            prev_indices
        ):

        dataset = load_dataset(dataset_name, split=split)
        unseen_data_idx = np.setdiff1d(range(len(dataset)), prev_indices)
        unseen_dataset = dataset.select(unseen_data_idx)

        unseen_dataloader = torch.utils.data.DataLoader(unseen_dataset, batch_size=args.batch_size, shuffle=False)
        _, _, unseen_losses = do_epoch(model, split, unseen_dataloader, tokenizer, optimizer, args, True)
        
        index_cutoff = int(len(dataset) * (1 / num_epochs))  # might have an off-by-one error but not that deep
        if anti == False:
            # all indexes with loss smaller than index_cutoff go at the start of the array
            selected_unseen_idx = np.argpartition(unseen_losses, index_cutoff)[:index_cutoff]
        else:
            if index_cutoff <= len(unseen_losses):
                selected_unseen_idx = np.argpartition(unseen_losses, -index_cutoff)[-index_cutoff:]
            else:
                selected_unseen_idx = np.arange(len(unseen_losses))
        new_indices = unseen_data_idx[selected_unseen_idx]

        self.indices_to_train = np.append(prev_indices, new_indices)
        self.curriculum_dataset = dataset.select(self.indices_to_train)

    def __len__(self):
        return len(self.curriculum_dataset)

    def __getitem__(self, idx):
        return self.curriculum_dataset[idx]
