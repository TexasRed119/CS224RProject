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
            anti,
            prev_indices,
            prev_losses=None,
            ref_model=None,
            is_brad=False
        ):

        if ref_model is not None or is_brad == True:
            dataset_dict = load_dataset("json", data_files=dataset_name)
            dataset = dataset_dict["train"]
        else:
            dataset = load_dataset(dataset_name, split=split)
        if cur_epoch + 1 == num_epochs:  # avoid calculating remaining losses
            self.indices_to_train = np.arange(len(dataset))
            self.curriculum_dataset = dataset
            return
        
        unseen_data_idx = np.setdiff1d(range(len(dataset)), prev_indices)
        if prev_losses is None:
            with torch.no_grad():
                unseen_dataset = dataset.select(unseen_data_idx)
                unseen_dataloader = torch.utils.data.DataLoader(unseen_dataset, batch_size=args.batch_size, shuffle=False)
                if ref_model is not None: 
                    _, _, self.unseen_losses = do_epoch(model, ref_model, split, unseen_dataloader, tokenizer, optimizer, args, scheduler=None, curriculum_init=True)
                else:
                    _, _, self.unseen_losses = do_epoch(model, split, unseen_dataloader, tokenizer, optimizer, args, scheduler=None, curriculum_init=True)
        else:
            self.unseen_losses = np.array(prev_losses)[unseen_data_idx]

        index_cutoff = int(len(dataset) * (1 / num_epochs))  # might have an off-by-one error but not that deep
        if anti == False:
            # all indexes with loss smaller than index_cutoff go at the start of the array
            selected_unseen_idx = np.argpartition(self.unseen_losses, index_cutoff)[:index_cutoff]
        else:
            if index_cutoff <= len(self.unseen_losses):
                selected_unseen_idx = np.argpartition(self.unseen_losses, -index_cutoff)[-index_cutoff:]
            else:
                selected_unseen_idx = np.arange(len(self.unseen_losses))
        new_indices = unseen_data_idx[selected_unseen_idx]

        self.indices_to_train = np.append(prev_indices, new_indices)
        self.curriculum_dataset = dataset.select(self.indices_to_train)

    def __len__(self):
        return len(self.curriculum_dataset)

    def __getitem__(self, idx):
        return self.curriculum_dataset[idx]
