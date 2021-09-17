import torch
import torch.nn as nn
from torch.nn.modules import dropout, linear
from transformers import AutoModel
from utils import *

class TransformerModel(nn.Module):
    def __init__(self, model='bert-base-cased', units=256, n_classes=2, rate=0.2):
        super(TransformerModel).__init__()
        # self.config = AutoConfig(model, num_labels=2, )
        self.model = AutoModel(
            model, 
            return_dict=False
        )
        self.linear = nn.Linear(self.model.config.hidden_size, units)
        self.pred = nn.Linear(units, n_classes)

        self.drop1 = nn.Dropout(rate)
        self.drop2 = nn.Dropout(rate)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    return_dict=False
                                    )
        linear_tr = self.linear(pooled_output)
        dropout_linear = self.drop1(linear_tr)
        linear_pred = self.pred(dropout_linear)
        # return self.drop2(linear_pred)
        return linear_pred
    
