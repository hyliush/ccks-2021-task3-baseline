import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel,BertModel
import torch
from collections import namedtuple
model_output=namedtuple('out',['loss','logits'])
class PointwiseMatching(BertPreTrainedModel):
    # 此处的 pretained_model 在本例中会被 ERNIE1.0 预训练模型初始化
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout( 0.1)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # module.allpy 迭代进入children，实际进入modules

        for n,m in self.named_children():
            if n=='classifier':
            # if m.__class__.__name__ == 'Linear':
            # if type(m) == nn.Linear
                if hasattr(self.config,'init'):
                    if self.config.init == 'normal':
                        nn.init.xavier_normal_(m.weight)
                    if self.config.init == 'uniform':
                        nn.init.xavier_uniform_(m.weight)
                else:
                    m.weight.fill_(1)
                m.bias.fill_(0)

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                add_graph=False,
                position_ids=None,
                labels=None,return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(input_ids, attention_mask,token_type_ids, position_ids)

        cls_embedding = outputs.get('pooler_output')
        #cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss=loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        else:
            loss = None

        if not return_dict or add_graph:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return {'loss':loss,'logits':logits}





