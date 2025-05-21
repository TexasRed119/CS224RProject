import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
BRADLEY_TERRY_DATASET = "HuggingFaceH4/ultrafeedback_binarized"
device = 'cuda'

class RewardModel(nn.Module):
    def __init__(self):
        super(RewardModel, self).__init__()

        self.base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
        hidden_dim = self.base_model.config.hidden_size
        self.output_layer = nn.Linear(hidden_dim, 1)

        parameters = list(self.base_model.parameters()) + list(self.output_layer.parameters())
        self.optimizer = torch.optim.AdamW(parameters)

    def forward(self, x):
        outputs = self.base_model(**x, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        return self.output_layer(last_hidden_state)

model = RewardModel()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
x = tokenizer(["hello", "world"], return_tensors='pt')
