from transformers import AutoModel
import torch
from torch import nn

from transformers import PreTrainedModel


class Model(PreTrainedModel):
    def __init__(self, config, labels_number, pretrained_model_name, tokenizer):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.labels_number = labels_number
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.bert.resize_token_embeddings(tokenizer.vocab_size)
        self.dropout = nn.Dropout(p=0.2)
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(self.bert.config.hidden_size, labels_number)
        self.linear.weight.data.uniform_(0.0, 1.0)

    def forward(self, input_ids):
        output = self.bert(input_ids=input_ids,
                           return_dict=False)[0]
        output = self.dropout(output)
        output = self.tanh(output)
        output = self.linear(output)
        output = output.squeeze(0)

        if len(output.shape) == 2:
            output = output.unsqueeze(0)

        cls_ids = torch.nonzero(input_ids == self.tokenizer.cls_token_id)
        filtered_logits = torch.zeros(cls_ids.shape[0], output.shape[2])

        for n in range(cls_ids.shape[0]):
            i, j = cls_ids[n]
            filtered_logits[n] = output[i, j, :]

        return filtered_logits
