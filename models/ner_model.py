import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import CRF
from models.bert import BertModel as MyBertModel
from torch.nn import CrossEntropyLoss
from losses.focal_loss import FocalLoss
from losses.label_smoothing import LabelSmoothingCrossEntropy
from losses.focal_loss import FocalLoss
from transformers import BertModel, BertPreTrainedModel


def get_attn_pad_mask(seq_q, seq_k, pad_id):
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    b_size, len_q = seq_q.size()
    b_size, len_k = seq_k.size()
    pad_attn_mask = (seq_k.data != pad_id).unsqueeze(1).long() # b_size x 1 x len_k
    return pad_attn_mask.expand(b_size, len_q, len_k)  # b_size x len_q x len_k


class BertSoftmaxForNer(nn.Module):
    """
    使用自己实现的Bert模型
    """
    def __init__(self, n_layers, d_model, d_ff, n_heads,
                 max_seq_len, vocab_size, pad_id, num_labels, loss_type, dropout=0.1):
        super(BertSoftmaxForNer, self).__init__()
        self.pad_id = pad_id
        self.num_labels = num_labels
        self.bert = MyBertModel(n_layers, d_model, d_ff, n_heads, max_seq_len, vocab_size, pad_id, dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_labels)
        self.loss_type = loss_type  # 使用哪种类型的损失函数
        # self.init_weights()   #todo 参数初始化

    def forward(self, input_ids, token_type_ids, labels, mask=None, return_attn=False):
        if mask is None:
            mask = get_attn_pad_mask(input_ids, input_ids, self.pad_id)
        else:
            input_len = mask.size(1)
            mask = mask.unsqueeze(1)    # [b_size x 1 x len]
            mask = mask.repeat(1, input_len, 1)
        # out:[b_size x len x d_model]
        out = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, mask=mask, return_attn=return_attn)
        out = self.dropout(out)
        logits = self.classifier(out)
        outputs = (logits,)
        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=self.pad_id)
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss(ignore_index=self.pad_id)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=self.pad_id)

            # 计算loss
            logits_ = logits.view(-1, self.num_labels)
            labels_ = labels.view(-1)
            loss = loss_fct(logits_, labels_)
            # Only keep active parts of the loss
            # if mask is not None:
            #     active_loss = mask.view(-1) == 1
            #     active_logits = logits.view(-1, self.num_labels)[active_loss]
            #     active_labels = labels.view(-1)[active_loss]
            #     loss = loss_fct(active_logits, active_labels)
            # else:
            #     loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)


class BertCrfForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCrfForNer, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,labels=None):
        outputs =self.bert(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions = logits, tags=labels, mask=attention_mask)
            outputs =(-1*loss,)+outputs
        return outputs # (loss), scores

class BertSoftmaxForNer_(BertPreTrainedModel):
    """
    使用transformers包中的Bert模型
    """
    def __init__(self, config):
        super(BertSoftmaxForNer_, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_type = config.loss_type
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,labels=None):
        outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss(ignore_index=0)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.contiguous().view(-1) == 1
                active_logits = logits.contiguous().view(-1, self.num_labels)[active_loss]
                active_labels = labels.contiguous().view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)


if __name__ == '__main__':
    input_ids = torch.randint(0,100,(4,10))
    token_type_ids = torch.randint(0,1,(4,10))
    model = BertSoftmaxForNer(n_layers=2, d_model=64, d_ff=256, n_heads=4,
                              max_seq_len=128, vocab_size=1000, pad_id=0, num_labels=3, loss_type='lsr', dropout=0.1)
    labels = torch.randint(0,3,(4,10))
    # model = BertModel(n_layers=2, d_model=64, d_ff=256, n_heads=4,
    #              max_seq_len=128, vocab_size=1000, pad_id=0, dropout=0.1)
    loss, logits = model(input_ids, token_type_ids, labels)
    print(loss)
