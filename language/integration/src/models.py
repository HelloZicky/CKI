"""Custom models for few-shot learning specific operations."""
import copy

import torch
import torch.nn as nn
import transformers
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertForSequenceClassification, BertModel, BertOnlyMLMHead
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification, RobertaModel, RobertaLMHead, RobertaClassificationHead
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import GPT2LMHeadModel

import logging

from tools import initializer

logger = logging.getLogger(__name__)

def resize_token_type_embeddings(model, new_num_types: int, random_segment: bool):
    """
    Resize the segment (token type) embeddings for BERT
    """
    if hasattr(model, 'bert'):
        old_token_type_embeddings = model.bert.embeddings.token_type_embeddings
    else:
        raise NotImplementedError
    new_token_type_embeddings = nn.Embedding(new_num_types, old_token_type_embeddings.weight.size(1))
    if not random_segment:
        new_token_type_embeddings.weight.data[:old_token_type_embeddings.weight.size(0)] = old_token_type_embeddings.weight.data

    model.config.type_vocab_size = new_num_types
    if hasattr(model, 'bert'):
        model.bert.embeddings.token_type_embeddings = new_token_type_embeddings
    else:
        raise NotImplementedError


class BertForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For label search.
        self.return_full_softmax = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        labels=None,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls(sequence_mask_output)

        # Exit early and only return mask logits.
        if self.return_full_softmax:
            if labels is not None:
                return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits) # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((loss,) + output) if loss is not None else output



class RobertaForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        
        self.classifier = None
        #torch.nn.Linear(config.hidden_size, config.num_labels)
        #RobertaClassificationHead(config)
        self.lm_head = RobertaLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For auto label search.
        self.return_full_softmax = None
        self.return_representation = None
        
        self.initial_parameters_copy = []#[ p.detach().clone() for p in self.roberta.parameters() ]

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask
        )

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        #print (sequence_output.shape)
        #print (mask_pos)
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]
        #print (sequence_mask_output.shape)
        sequence_CLS_output  = sequence_output[torch.arange(sequence_output.size(0)), 0]

        if self.model_args.use_lm_head:
            # Logits over vocabulary tokens
            if self.return_representation:
                return sequence_mask_output
            
            
            prediction_mask_scores = self.lm_head(sequence_mask_output)
            #print (prediction_mask_scores.shape)
            
            
            # Exit early and only return mask logits.
            if self.return_full_softmax:
                if labels is not None:
                    return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
                return prediction_mask_scores

            # Return logits for each label
            logits = []
            for label_id in range(len(self.label_word_list)):
                logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
            logits = torch.cat(logits, -1)
        elif self.model_args.use_CLS_linearhead:
            logits = self.classifier(sequence_CLS_output)
       
            
        
        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits) # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            if self.model_args.l1_reg != 0.0:
                l1_norm = sum(torch.sum(torch.abs(p - q.to('cuda'))) for p, q in zip( self.roberta.parameters(), self.initial_parameters_copy ) )
                loss += self.model_args.l1_reg * l1_norm
            
        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        
        
        return ((loss,) + output) if loss is not None else output

    
    
class GPTPromptFinetuning(GPT2LMHeadModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.gpt = GPT2LMHeadModel(config)
        
        self.classifier = None
        #torch.nn.Linear(config.hidden_size, config.num_labels)
        #RobertaClassificationHead(config)
        #self.lm_head = RobertaLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For auto label search.
        self.return_full_softmax = None
        self.return_representation = None
        
        self.initial_parameters_copy = []#[ p.detach().clone() for p in self.roberta.parameters() ]

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.gpt(
            input_ids,
            attention_mask=attention_mask
        )

        
        
        
        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        #print (sequence_output.shape)
        #print (mask_pos)
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]
        #print (sequence_mask_output.shape)
        sequence_CLS_output  = sequence_output[torch.arange(sequence_output.size(0)), 0]

        if self.model_args.use_lm_head:
            # Logits over vocabulary tokens
            if self.return_representation:
                return sequence_mask_output
            
            
            prediction_mask_scores = self.lm_head(sequence_mask_output)
            #print (prediction_mask_scores.shape)
            
            
            # Exit early and only return mask logits.
            if self.return_full_softmax:
                if labels is not None:
                    return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
                return prediction_mask_scores

            # Return logits for each label
            logits = []
            for label_id in range(len(self.label_word_list)):
                logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
            logits = torch.cat(logits, -1)
        elif self.model_args.use_CLS_linearhead:
            logits = self.classifier(sequence_CLS_output)
       
            
        
        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits) # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            if self.model_args.l1_reg != 0.0:
                l1_norm = sum(torch.sum(torch.abs(p - q.to('cuda'))) for p, q in zip( self.roberta.parameters(), self.initial_parameters_copy ) )
                loss += self.model_args.l1_reg * l1_norm
            
        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        
        
        return ((loss,) + output) if loss is not None else output

class Linear(torch.nn.Module):
    def __init__(self, in_dimension, out_dimension, bias):
        super(Linear, self).__init__()
        self.net = torch.nn.Linear(in_dimension, out_dimension, bias)
        initializer.default_weight_init(self.net.weight)
        if bias:
            # initializer.default_weight_init(self.net.bias)
            initializer.default_bias_init(self.net.bias)

    def __setitem__(self, k, v):
        self.k = v

    def forward(self, x):
        return self.net(x)

# v1
class UnitingModelForNLP_v1(torch.nn.Module):
    def __init__(self, config, model_type):
        super(UnitingModelForNLP, self).__init__()

        self.module = []

        if model_type == "roberta":
            self.dense_assess = nn.Linear(config.hidden_size, config.hidden_size)
            initializer.default_weight_init(self.dense_assess.weight)
            initializer.default_bias_init(self.dense_assess.bias)
            self.module.append(self.dense_assess)

            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            self.dropout = nn.Dropout(classifier_dropout)

        self.out_assess = nn.Linear(config.hidden_size, config.hidden_size)
        initializer.default_weight_init(self.out_assess.weight)
        initializer.default_bias_init(self.out_assess.bias)
        self.module.append(self.out_assess)

        self.model_type = model_type
        self.a = 0.4
        self.c = 500


    def forward(self, x, base_param, graft_param):
        if self.model_type == "roberta":
            base_param_weight = ["dense.weight", "out_proj.weight"]
            base_param_bias = ["dense.bias", "out_proj.bias"]
        elif self.model_type == "bert":
            base_param_weight = ["weight"]
            base_param_bias = ["bias"]

        def entropy(x, n=10):
            x = x.reshape(-1)
            scale = (x.max() - x.min()) / n
            entropy = torch.tensor(0.).to(x.device)
            for i in range(n):
                p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale), dtype=torch.float) / len(x)
                if p != 0:
                    entropy -= p * torch.log(p)
            return entropy

        if self.model_type == "roberta":
            x = x[:, 0, :]
        for i in range(len(base_param_weight)):
            param_i = base_param[base_param_weight[i]]
            param_j = graft_param[base_param_weight[i]]
            sigmoid = nn.Sigmoid()
            param_direction = torch.abs(param_i - param_j)
            param_direction = self.module[i](param_direction)
            w_local = sigmoid(param_direction)
            # w_local = param_direction
            w_global = self.a / torch.pi * torch.arctan(self.c * (entropy(param_i) - entropy(param_j))) + 0.5
            w = w_global * w_local + (1 - w_global)

            param_uniting = param_i * w + param_j * (1 - w)

            if self.model_type == "roberta":
                x = self.dropout(x)
            x = torch.matmul(x, param_uniting.transpose(0, 1)) + base_param[base_param_bias[i]]

            if i + 1 != len(base_param_weight):
                x = torch.tanh(x)
            else:
                return x

    def load_param(self, base_model, base_param, graft_param):
        new_param = copy.deepcopy(base_param)

        if self.model_type == "roberta":
            base_param_weight = ["dense.weight", "out_proj.weight"]
            base_param_bias = ["dense.bias", "out_proj.bias"]

        def entropy(x, n=10):
            x = x.reshape(-1)
            scale = (x.max() - x.min()) / n
            entropy = torch.tensor(0.).to(x.device)
            for i in range(n):
                p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale), dtype=torch.float) / len(x)
                if p != 0:
                    entropy -= p * torch.log(p)
            return entropy

        with torch.no_grad():
            # base_param = base_model.classifier.state_dict()
            for i in range(len(base_param_weight)):
                param_i = base_param[base_param_weight[i]]
                param_j = graft_param[base_param_weight[i]]

                sigmoid = nn.Sigmoid()
                param_direction = param_i - param_j
                param_direction = self.module[i](param_direction)
                w_local = sigmoid(param_direction)
                w_global = self.a / torch.pi * torch.arctan(self.c * (entropy(param_i) - entropy(param_j))) + 0.5

                # v1
                w = w_global * w_local + (1 - w_global)
                param_uniting = param_i * w + param_j * (1 - w)

                # w_base = 1 - w_global * torch.exp(-w_global * w_local)
                # w_graft = 1 - w_global * torch.exp(w_global * w_local - 1)
                #
                # w = torch.stack([w_base, w_graft], dim=-1)
                # softmax = nn.Softmax(dim=-1)
                # w = softmax(w)
                # w_base, w_graft = torch.split(w, 1, dim=-1)
                # w_base = w_base.squeeze(dim=-1)
                # w_graft = w_graft.squeeze(dim=-1)
                #
                # param_uniting = param_i * w_base + param_j * w_graft

                new_param[base_param_weight[i]] = param_uniting

            base_model.classifier.load_state_dict(new_param)
        return base_model

# v2: 差值*5
class UnitingModelForNLP_v2(torch.nn.Module):
        def __init__(self, config, model_type):
            super(UnitingModelForNLP, self).__init__()

            self.module = []

            self.dense_assess = nn.Linear(config.hidden_size, config.hidden_size)
            initializer.default_weight_init(self.dense_assess.weight)
            initializer.default_bias_init(self.dense_assess.bias)
            self.module.append(self.dense_assess)

            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            self.dropout = nn.Dropout(classifier_dropout)

            self.out_assess = nn.Linear(config.hidden_size, config.hidden_size)
            initializer.default_weight_init(self.out_assess.weight)
            initializer.default_bias_init(self.out_assess.bias)
            self.module.append(self.out_assess)

            self.model_type = model_type
            self.a = 0.4
            self.c = 500

        def forward(self, x, base_param, graft_param):
            if self.model_type == "roberta":
                base_param_weight = ["dense.weight", "out_proj.weight"]
                base_param_bias = ["dense.bias", "out_proj.bias"]

            def entropy(x, n=10):
                x = x.reshape(-1)
                scale = (x.max() - x.min()) / n
                entropy = torch.tensor(0.).to(x.device)
                for i in range(n):
                    p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale),
                                  dtype=torch.float) / len(x)
                    if p != 0:
                        entropy -= p * torch.log(p)
                return entropy

            x = x[:, 0, :]
            for i in range(len(base_param_weight)):
                param_i = base_param[base_param_weight[i]]
                param_j = graft_param[base_param_weight[i]]
                sigmoid = nn.Sigmoid()
                param_direction = torch.abs(param_i - param_j)
                param_direction = self.module[i](param_direction) * 5
                w_local = sigmoid(param_direction)
                # w_local = param_direction
                w_global = self.a / torch.pi * torch.arctan(self.c * (entropy(param_i) - entropy(param_j))) + 0.5
                w = w_global * w_local + (1 - w_global)

                param_uniting = param_i * w + param_j * (1 - w)

                x = self.dropout(x)
                x = torch.matmul(x, param_uniting.transpose(0, 1)) + base_param[base_param_bias[i]]

                if i + 1 != len(base_param_weight):
                    x = torch.tanh(x)
                else:
                    return x

        def load_param(self, base_model, base_param, graft_param):
            new_param = copy.deepcopy(base_param)

            if self.model_type == "roberta":
                base_param_weight = ["dense.weight", "out_proj.weight"]
                base_param_bias = ["dense.bias", "out_proj.bias"]

            def entropy(x, n=10):
                x = x.reshape(-1)
                scale = (x.max() - x.min()) / n
                entropy = torch.tensor(0.).to(x.device)
                for i in range(n):
                    p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale),
                                  dtype=torch.float) / len(x)
                    if p != 0:
                        entropy -= p * torch.log(p)
                return entropy

            with torch.no_grad():
                # base_param = base_model.classifier.state_dict()
                for i in range(len(base_param_weight)):
                    param_i = base_param[base_param_weight[i]]
                    param_j = graft_param[base_param_weight[i]]

                    sigmoid = nn.Sigmoid()
                    param_direction = param_i - param_j
                    param_direction = self.module[i](param_direction) * 5
                    w_local = sigmoid(param_direction)
                    w_global = self.a / torch.pi * torch.arctan(self.c * (entropy(param_i) - entropy(param_j))) + 0.5

                    # v1
                    w = w_global * w_local + (1 - w_global)
                    param_uniting = param_i * w + param_j * (1 - w)

                    new_param[base_param_weight[i]] = param_uniting

                base_model.classifier.load_state_dict(new_param)
            return base_model

# v2b: 差值*5
class UnitingModelForNLP_v2b(torch.nn.Module):
        def __init__(self, config, model_type):
            super(UnitingModelForNLP, self).__init__()

            self.module = []

            self.out_assess = nn.Linear(config.hidden_size, config.hidden_size)
            initializer.default_weight_init(self.out_assess.weight)
            initializer.default_bias_init(self.out_assess.bias)
            self.module.append(self.out_assess)

            if model_type == "roberta":
                classifier_dropout = (
                    config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
                )
                self.dropout = nn.Dropout(classifier_dropout)

            self.model_type = model_type
            self.a = 0.4
            self.c = 500

        def forward(self, x, base_param, graft_param):
            if self.model_type == "roberta":
                base_param_weight = ["dense.weight", "out_proj.weight"]
                base_param_bias = ["dense.bias", "out_proj.bias"]
            elif self.model_type == "bert":
                base_param_weight = ["weight"]
                base_param_bias = ["bias"]

            def entropy(x, n=10):
                x = x.reshape(-1)
                scale = (x.max() - x.min()) / n
                entropy = torch.tensor(0.).to(x.device)
                for i in range(n):
                    p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale),
                                  dtype=torch.float) / len(x)
                    if p != 0:
                        entropy -= p * torch.log(p)
                return entropy


            if self.model_type == "roberta":
                x = x[:, 0, :]

                param_i = base_param["out_proj.weight"]
                param_j = graft_param["out_proj.weight"]
            elif self.model_type == "bert":
                param_i = base_param["weight"]
                param_j = graft_param["weight"]
            sigmoid = nn.Sigmoid()
            param_direction = torch.abs(param_i - param_j)
            param_direction = self.out_assess(param_direction) * 5
            w_local = sigmoid(param_direction)
            # w_local = param_direction
            w_global = self.a / torch.pi * torch.arctan(self.c * (entropy(param_i) - entropy(param_j))) + 0.5
            w = w_global * w_local + (1 - w_global)

            out_param_uniting = param_i * w + param_j * (1 - w)

            if self.model_type == "roberta":
                x = self.dropout(x)
                x = torch.matmul(x, base_param["dense.weight"].transpose(0, 1)) + base_param["dense.bias"]
                x = torch.tanh(x)
                x = self.dropout(x)
                x = torch.matmul(x, out_param_uniting.transpose(0, 1)) + base_param["out_proj.bias"]
            elif self.model_type == "bert":
                x = torch.matmul(x, out_param_uniting.transpose(0, 1)) + base_param["bias"]

            return x

        def load_param(self, base_model, base_param, graft_param):
            new_param = copy.deepcopy(base_param)

            if self.model_type == "roberta":
                base_param_weight = ["dense.weight", "out_proj.weight"]
                base_param_bias = ["dense.bias", "out_proj.bias"]

            def entropy(x, n=10):
                x = x.reshape(-1)
                scale = (x.max() - x.min()) / n
                entropy = torch.tensor(0.).to(x.device)
                for i in range(n):
                    p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale),
                                  dtype=torch.float) / len(x)
                    if p != 0:
                        entropy -= p * torch.log(p)
                return entropy

            with torch.no_grad():
                if self.model_type == "roberta":
                    param_i = base_param["out_proj.weight"]
                    param_j = graft_param["out_proj.weight"]
                elif self.model_type == "bert":
                    param_i = base_param["weight"]
                    param_j = graft_param["weight"]
                sigmoid = nn.Sigmoid()
                param_direction = torch.abs(param_i - param_j)
                param_direction = self.out_assess(param_direction) * 5
                w_local = sigmoid(param_direction)
                # w_local = param_direction
                w_global = self.a / torch.pi * torch.arctan(self.c * (entropy(param_i) - entropy(param_j))) + 0.5
                w = w_global * w_local + (1 - w_global)

                out_param_uniting = param_i * w + param_j * (1 - w)

                if self.model_type == "roberta":
                    new_param["out_proj.weight"] = out_param_uniting
                elif self.model_type == "bert":
                    new_param["weight"] = out_param_uniting

                base_model.classifier.load_state_dict(new_param)
            return base_model

# v3: 只拼接dense层
class UnitingModelForNLP_v3(torch.nn.Module):
        def __init__(self, config, model_type):
            super(UnitingModelForNLP, self).__init__()

            self.module = []

            self.dense_assess = nn.Linear(config.hidden_size, config.hidden_size)
            initializer.default_weight_init(self.dense_assess.weight)
            initializer.default_bias_init(self.dense_assess.bias)
            self.module.append(self.dense_assess)

            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            self.dropout = nn.Dropout(classifier_dropout)

            self.model_type = model_type
            self.a = 0.4
            self.c = 500

        def forward(self, x, base_param, graft_param):
            if self.model_type == "roberta":
                base_param_weight = ["dense.weight", "out_proj.weight"]
                base_param_bias = ["dense.bias", "out_proj.bias"]

            def entropy(x, n=10):
                x = x.reshape(-1)
                scale = (x.max() - x.min()) / n
                entropy = torch.tensor(0.).to(x.device)
                for i in range(n):
                    p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale),
                                  dtype=torch.float) / len(x)
                    if p != 0:
                        entropy -= p * torch.log(p)
                return entropy

            x = x[:, 0, :]

            param_i = base_param["dense.weight"]
            param_j = graft_param["dense.weight"]
            sigmoid = nn.Sigmoid()
            param_direction = torch.abs(param_i - param_j)
            param_direction = self.dense_assess(param_direction)
            w_local = sigmoid(param_direction)
            # w_local = param_direction
            w_global = self.a / torch.pi * torch.arctan(self.c * (entropy(param_i) - entropy(param_j))) + 0.5
            w = w_global * w_local + (1 - w_global)

            dense_param_uniting = param_i * w + param_j * (1 - w)

            x = self.dropout(x)
            x = torch.matmul(x, dense_param_uniting.transpose(0, 1)) + base_param["dense.bias"]
            x = torch.tanh(x)
            x = self.dropout(x)
            x = torch.matmul(x, base_param["out_proj.weight"].transpose(0, 1)) + base_param["out_proj.bias"]

            return x

        def load_param(self, base_model, base_param, graft_param):
            new_param = copy.deepcopy(base_param)

            if self.model_type == "roberta":
                base_param_weight = ["dense.weight", "out_proj.weight"]
                base_param_bias = ["dense.bias", "out_proj.bias"]

            def entropy(x, n=10):
                x = x.reshape(-1)
                scale = (x.max() - x.min()) / n
                entropy = torch.tensor(0.).to(x.device)
                for i in range(n):
                    p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale),
                                  dtype=torch.float) / len(x)
                    if p != 0:
                        entropy -= p * torch.log(p)
                return entropy

            with torch.no_grad():
                param_i = base_param["dense.weight"]
                param_j = graft_param["dense.weight"]
                sigmoid = nn.Sigmoid()
                param_direction = torch.abs(param_i - param_j)
                param_direction = self.dense_assess(param_direction)
                w_local = sigmoid(param_direction)
                # w_local = param_direction
                w_global = self.a / torch.pi * torch.arctan(self.c * (entropy(param_i) - entropy(param_j))) + 0.5
                w = w_global * w_local + (1 - w_global)

                dense_param_uniting = param_i * w + param_j * (1 - w)

                new_param["dense.weight"] = dense_param_uniting

                base_model.classifier.load_state_dict(new_param)
            return base_model

# v4: 只拼接dense层，差值*10
class UnitingModelForNLP_v4(torch.nn.Module):
        def __init__(self, config, model_type):
            super(UnitingModelForNLP, self).__init__()

            self.module = []

            self.dense_assess = nn.Linear(config.hidden_size, config.hidden_size)
            initializer.default_weight_init(self.dense_assess.weight)
            initializer.default_bias_init(self.dense_assess.bias)
            self.module.append(self.dense_assess)

            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            self.dropout = nn.Dropout(classifier_dropout)

            self.model_type = model_type
            self.a = 0.4
            self.c = 500

        def forward(self, x, base_param, graft_param):
            if self.model_type == "roberta":
                base_param_weight = ["dense.weight", "out_proj.weight"]
                base_param_bias = ["dense.bias", "out_proj.bias"]

            def entropy(x, n=10):
                x = x.reshape(-1)
                scale = (x.max() - x.min()) / n
                entropy = torch.tensor(0.).to(x.device)
                for i in range(n):
                    p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale),
                                  dtype=torch.float) / len(x)
                    if p != 0:
                        entropy -= p * torch.log(p)
                return entropy

            x = x[:, 0, :]

            param_i = base_param["dense.weight"]
            param_j = graft_param["dense.weight"]
            sigmoid = nn.Sigmoid()
            param_direction = torch.abs(param_i - param_j)
            param_direction = self.dense_assess(param_direction) * 10
            w_local = sigmoid(param_direction)
            # w_local = param_direction
            w_global = self.a / torch.pi * torch.arctan(self.c * (entropy(param_i) - entropy(param_j))) + 0.5
            w = w_global * w_local + (1 - w_global)

            dense_param_uniting = param_i * w + param_j * (1 - w)

            x = self.dropout(x)
            x = torch.matmul(x, dense_param_uniting.transpose(0, 1)) + base_param["dense.bias"]
            x = torch.tanh(x)
            x = self.dropout(x)
            x = torch.matmul(x, base_param["out_proj.weight"].transpose(0, 1)) + base_param["out_proj.bias"]

            return x

        def load_param(self, base_model, base_param, graft_param):
            new_param = copy.deepcopy(base_param)

            if self.model_type == "roberta":
                base_param_weight = ["dense.weight", "out_proj.weight"]
                base_param_bias = ["dense.bias", "out_proj.bias"]

            def entropy(x, n=10):
                x = x.reshape(-1)
                scale = (x.max() - x.min()) / n
                entropy = torch.tensor(0.).to(x.device)
                for i in range(n):
                    p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale),
                                  dtype=torch.float) / len(x)
                    if p != 0:
                        entropy -= p * torch.log(p)
                return entropy

            with torch.no_grad():
                param_i = base_param["dense.weight"]
                param_j = graft_param["dense.weight"]
                sigmoid = nn.Sigmoid()
                param_direction = torch.abs(param_i - param_j)
                param_direction = self.dense_assess(param_direction) * 10
                w_local = sigmoid(param_direction)
                # w_local = param_direction
                w_global = self.a / torch.pi * torch.arctan(self.c * (entropy(param_i) - entropy(param_j))) + 0.5
                w = w_global * w_local + (1 - w_global)

                dense_param_uniting = param_i * w + param_j * (1 - w)

                new_param["dense.weight"] = dense_param_uniting

                base_model.classifier.load_state_dict(new_param)
            return base_model

# v4b: 差值*10
class UnitingModelForNLP_v4b(torch.nn.Module):
        def __init__(self, config, model_type):
            super(UnitingModelForNLP, self).__init__()

            self.module = []

            self.out_assess = nn.Linear(config.hidden_size, config.hidden_size)
            initializer.default_weight_init(self.out_assess.weight)
            initializer.default_bias_init(self.out_assess.bias)
            self.module.append(self.out_assess)

            if model_type == "roberta":
                classifier_dropout = (
                    config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
                )
                self.dropout = nn.Dropout(classifier_dropout)

            self.model_type = model_type
            self.a = 0.4
            self.c = 500

        def forward(self, x, base_param, graft_param):
            if self.model_type == "roberta":
                base_param_weight = ["dense.weight", "out_proj.weight"]
                base_param_bias = ["dense.bias", "out_proj.bias"]
            elif self.model_type == "bert":
                base_param_weight = ["weight"]
                base_param_bias = ["bias"]

            def entropy(x, n=10):
                x = x.reshape(-1)
                scale = (x.max() - x.min()) / n
                entropy = torch.tensor(0.).to(x.device)
                for i in range(n):
                    p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale),
                                  dtype=torch.float) / len(x)
                    if p != 0:
                        entropy -= p * torch.log(p)
                return entropy


            if self.model_type == "roberta":
                x = x[:, 0, :]

                param_i = base_param["out_proj.weight"]
                param_j = graft_param["out_proj.weight"]
            elif self.model_type == "bert":
                param_i = base_param["weight"]
                param_j = graft_param["weight"]
            sigmoid = nn.Sigmoid()
            param_direction = torch.abs(param_i - param_j)
            param_direction = self.out_assess(param_direction) * 10
            w_local = sigmoid(param_direction)
            # w_local = param_direction
            w_global = self.a / torch.pi * torch.arctan(self.c * (entropy(param_i) - entropy(param_j))) + 0.5
            w = w_global * w_local + (1 - w_global)

            out_param_uniting = param_i * w + param_j * (1 - w)

            if self.model_type == "roberta":
                x = self.dropout(x)
                x = torch.matmul(x, base_param["dense.weight"].transpose(0, 1)) + base_param["dense.bias"]
                x = torch.tanh(x)
                x = self.dropout(x)
                x = torch.matmul(x, out_param_uniting.transpose(0, 1)) + base_param["out_proj.bias"]
            elif self.model_type == "bert":
                x = torch.matmul(x, out_param_uniting.transpose(0, 1)) + base_param["bias"]

            return x

        def load_param(self, base_model, base_param, graft_param):
            new_param = copy.deepcopy(base_param)

            if self.model_type == "roberta":
                base_param_weight = ["dense.weight", "out_proj.weight"]
                base_param_bias = ["dense.bias", "out_proj.bias"]

            def entropy(x, n=10):
                x = x.reshape(-1)
                scale = (x.max() - x.min()) / n
                entropy = torch.tensor(0.).to(x.device)
                for i in range(n):
                    p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale),
                                  dtype=torch.float) / len(x)
                    if p != 0:
                        entropy -= p * torch.log(p)
                return entropy

            with torch.no_grad():
                if self.model_type == "roberta":
                    param_i = base_param["out_proj.weight"]
                    param_j = graft_param["out_proj.weight"]
                elif self.model_type == "bert":
                    param_i = base_param["weight"]
                    param_j = graft_param["weight"]
                sigmoid = nn.Sigmoid()
                param_direction = torch.abs(param_i - param_j)
                param_direction = self.out_assess(param_direction) * 10
                w_local = sigmoid(param_direction)
                # w_local = param_direction
                w_global = self.a / torch.pi * torch.arctan(self.c * (entropy(param_i) - entropy(param_j))) + 0.5
                w = w_global * w_local + (1 - w_global)

                out_param_uniting = param_i * w + param_j * (1 - w)

                if self.model_type == "roberta":
                    new_param["out_proj.weight"] = out_param_uniting
                elif self.model_type == "bert":
                    new_param["weight"] = out_param_uniting

                base_model.classifier.load_state_dict(new_param)
            return base_model

# v5: 只拼接out_proj层(bert & roberta)
class UnitingModelForNLP_v5(torch.nn.Module):
        def __init__(self, config, model_type):
            super(UnitingModelForNLP, self).__init__()

            self.module = []

            self.out_assess = nn.Linear(config.hidden_size, config.hidden_size)
            initializer.default_weight_init(self.out_assess.weight)
            initializer.default_bias_init(self.out_assess.bias)
            self.module.append(self.out_assess)

            if model_type == "roberta":
                classifier_dropout = (
                    config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
                )
                self.dropout = nn.Dropout(classifier_dropout)

            self.model_type = model_type
            self.a = 0.4
            self.c = 500

        def forward(self, x, base_param, graft_param):
            if self.model_type == "roberta":
                base_param_weight = ["dense.weight", "out_proj.weight"]
                base_param_bias = ["dense.bias", "out_proj.bias"]
            elif self.model_type == "bert":
                base_param_weight = ["weight"]
                base_param_bias = ["bias"]

            def entropy(x, n=10):
                x = x.reshape(-1)
                scale = (x.max() - x.min()) / n
                entropy = torch.tensor(0.).to(x.device)
                for i in range(n):
                    p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale),
                                  dtype=torch.float) / len(x)
                    if p != 0:
                        entropy -= p * torch.log(p)
                return entropy


            if self.model_type == "roberta":
                x = x[:, 0, :]

                param_i = base_param["out_proj.weight"]
                param_j = graft_param["out_proj.weight"]
            elif self.model_type == "bert":
                param_i = base_param["weight"]
                param_j = graft_param["weight"]
            sigmoid = nn.Sigmoid()
            param_direction = torch.abs(param_i - param_j)
            param_direction = self.out_assess(param_direction)
            w_local = sigmoid(param_direction)
            # w_local = param_direction
            w_global = self.a / torch.pi * torch.arctan(self.c * (entropy(param_i) - entropy(param_j))) + 0.5
            w = w_global * w_local + (1 - w_global)

            out_param_uniting = param_i * w + param_j * (1 - w)

            if self.model_type == "roberta":
                x = self.dropout(x)
                x = torch.matmul(x, base_param["dense.weight"].transpose(0, 1)) + base_param["dense.bias"]
                x = torch.tanh(x)
                x = self.dropout(x)
                x = torch.matmul(x, out_param_uniting.transpose(0, 1)) + base_param["out_proj.bias"]
            elif self.model_type == "bert":
                x = torch.matmul(x, out_param_uniting.transpose(0, 1)) + base_param["bias"]

            return x

        def load_param(self, base_model, base_param, graft_param):
            new_param = copy.deepcopy(base_param)

            if self.model_type == "roberta":
                base_param_weight = ["dense.weight", "out_proj.weight"]
                base_param_bias = ["dense.bias", "out_proj.bias"]

            def entropy(x, n=10):
                x = x.reshape(-1)
                scale = (x.max() - x.min()) / n
                entropy = torch.tensor(0.).to(x.device)
                for i in range(n):
                    p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale),
                                  dtype=torch.float) / len(x)
                    if p != 0:
                        entropy -= p * torch.log(p)
                return entropy

            with torch.no_grad():
                if self.model_type == "roberta":
                    param_i = base_param["out_proj.weight"]
                    param_j = graft_param["out_proj.weight"]
                elif self.model_type == "bert":
                    param_i = base_param["weight"]
                    param_j = graft_param["weight"]
                sigmoid = nn.Sigmoid()
                param_direction = torch.abs(param_i - param_j)
                param_direction = self.out_assess(param_direction)
                w_local = sigmoid(param_direction)
                # w_local = param_direction
                w_global = self.a / torch.pi * torch.arctan(self.c * (entropy(param_i) - entropy(param_j))) + 0.5
                w = w_global * w_local + (1 - w_global)

                out_param_uniting = param_i * w + param_j * (1 - w)

                if self.model_type == "roberta":
                    new_param["out_proj.weight"] = out_param_uniting
                elif self.model_type == "bert":
                    new_param["weight"] = out_param_uniting

                base_model.classifier.load_state_dict(new_param)
            return base_model

# v6b: hard fusion
class UnitingModelForNLP_v6b(torch.nn.Module):
        def __init__(self, config, model_type):
            super(UnitingModelForNLP, self).__init__()

            self.module = []

            self.out_assess = nn.Linear(config.hidden_size, config.hidden_size)
            initializer.default_weight_init(self.out_assess.weight)
            initializer.default_bias_init(self.out_assess.bias)
            self.module.append(self.out_assess)

            if model_type == "roberta":
                classifier_dropout = (
                    config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
                )
                self.dropout = nn.Dropout(classifier_dropout)

            self.model_type = model_type
            self.a = 0.4
            self.c = 500

        def forward(self, x, base_param, graft_param):
            if self.model_type == "roberta":
                base_param_weight = ["dense.weight", "out_proj.weight"]
                base_param_bias = ["dense.bias", "out_proj.bias"]
            elif self.model_type == "bert":
                base_param_weight = ["weight"]
                base_param_bias = ["bias"]

            def entropy(x, n=10):
                x = x.reshape(-1)
                scale = (x.max() - x.min()) / n
                entropy = torch.tensor(0.).to(x.device)
                for i in range(n):
                    p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale),
                                  dtype=torch.float) / len(x)
                    if p != 0:
                        entropy -= p * torch.log(p)
                return entropy


            if self.model_type == "roberta":
                x = x[:, 0, :]

                param_i = base_param["out_proj.weight"]
                param_j = graft_param["out_proj.weight"]
            elif self.model_type == "bert":
                param_i = base_param["weight"]
                param_j = graft_param["weight"]
            sigmoid = nn.Sigmoid()
            param_direction = torch.abs(param_i - param_j)
            param_direction = self.out_assess(param_direction)
            w_local = sigmoid(param_direction)
            # w_local = param_direction
            w_global = self.a / torch.pi * torch.arctan(self.c * (entropy(param_i) - entropy(param_j))) + 0.5
            w_i = w_global * w_local + (1 - w_global)
            w_j = 1 - w_i

            param_uniting = torch.where(w_i > w_j, param_i, param_j)

            if self.model_type == "roberta":
                x = self.dropout(x)
                x = torch.matmul(x, base_param["dense.weight"].transpose(0, 1)) + base_param["dense.bias"]
                x = torch.tanh(x)
                x = self.dropout(x)
                x = torch.matmul(x, param_uniting.transpose(0, 1)) + base_param["out_proj.bias"]
            elif self.model_type == "bert":
                x = torch.matmul(x, param_uniting.transpose(0, 1)) + base_param["bias"]

            return x

        def load_param(self, base_model, base_param, graft_param):
            new_param = copy.deepcopy(base_param)

            if self.model_type == "roberta":
                base_param_weight = ["dense.weight", "out_proj.weight"]
                base_param_bias = ["dense.bias", "out_proj.bias"]

            def entropy(x, n=10):
                x = x.reshape(-1)
                scale = (x.max() - x.min()) / n
                entropy = torch.tensor(0.).to(x.device)
                for i in range(n):
                    p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale),
                                  dtype=torch.float) / len(x)
                    if p != 0:
                        entropy -= p * torch.log(p)
                return entropy

            with torch.no_grad():
                if self.model_type == "roberta":
                    param_i = base_param["out_proj.weight"]
                    param_j = graft_param["out_proj.weight"]
                elif self.model_type == "bert":
                    param_i = base_param["weight"]
                    param_j = graft_param["weight"]
                sigmoid = nn.Sigmoid()
                param_direction = torch.abs(param_i - param_j)
                param_direction = self.out_assess(param_direction)
                w_local = sigmoid(param_direction)
                # w_local = param_direction
                w_global = self.a / torch.pi * torch.arctan(self.c * (entropy(param_i) - entropy(param_j))) + 0.5
                w_i = w_global * w_local + (1 - w_global)
                w_j = 1 - w_i

                param_uniting = torch.where(w_i > w_j, param_i, param_j)

                if self.model_type == "roberta":
                    new_param["out_proj.weight"] = param_uniting
                elif self.model_type == "bert":
                    new_param["weight"] = param_uniting

                base_model.classifier.load_state_dict(new_param)
            return base_model

# v7b: local
class UnitingModelForNLP_v7b(torch.nn.Module):
        def __init__(self, config, model_type):
            super(UnitingModelForNLP, self).__init__()

            self.module = []

            self.out_assess = nn.Linear(config.hidden_size, config.hidden_size)
            initializer.default_weight_init(self.out_assess.weight)
            initializer.default_bias_init(self.out_assess.bias)
            self.module.append(self.out_assess)

            if model_type == "roberta":
                classifier_dropout = (
                    config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
                )
                self.dropout = nn.Dropout(classifier_dropout)

            self.model_type = model_type
            self.a = 0.4
            self.c = 500

        def forward(self, x, base_param, graft_param):
            if self.model_type == "roberta":
                base_param_weight = ["dense.weight", "out_proj.weight"]
                base_param_bias = ["dense.bias", "out_proj.bias"]
            elif self.model_type == "bert":
                base_param_weight = ["weight"]
                base_param_bias = ["bias"]

            def entropy(x, n=10):
                x = x.reshape(-1)
                scale = (x.max() - x.min()) / n
                entropy = torch.tensor(0.).to(x.device)
                for i in range(n):
                    p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale),
                                  dtype=torch.float) / len(x)
                    if p != 0:
                        entropy -= p * torch.log(p)
                return entropy


            if self.model_type == "roberta":
                x = x[:, 0, :]

                param_i = base_param["out_proj.weight"]
                param_j = graft_param["out_proj.weight"]
            elif self.model_type == "bert":
                param_i = base_param["weight"]
                param_j = graft_param["weight"]
            sigmoid = nn.Sigmoid()
            param_direction = torch.abs(param_i - param_j)
            param_direction = self.out_assess(param_direction)
            w_local = sigmoid(param_direction)

            param_uniting = param_i * w_local + param_j * (1 - w_local)

            if self.model_type == "roberta":
                x = self.dropout(x)
                x = torch.matmul(x, base_param["dense.weight"].transpose(0, 1)) + base_param["dense.bias"]
                x = torch.tanh(x)
                x = self.dropout(x)
                x = torch.matmul(x, param_uniting.transpose(0, 1)) + base_param["out_proj.bias"]
            elif self.model_type == "bert":
                x = torch.matmul(x, param_uniting.transpose(0, 1)) + base_param["bias"]

            return x

        def load_param(self, base_model, base_param, graft_param):
            new_param = copy.deepcopy(base_param)

            if self.model_type == "roberta":
                base_param_weight = ["dense.weight", "out_proj.weight"]
                base_param_bias = ["dense.bias", "out_proj.bias"]

            def entropy(x, n=10):
                x = x.reshape(-1)
                scale = (x.max() - x.min()) / n
                entropy = torch.tensor(0.).to(x.device)
                for i in range(n):
                    p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale),
                                  dtype=torch.float) / len(x)
                    if p != 0:
                        entropy -= p * torch.log(p)
                return entropy

            with torch.no_grad():
                if self.model_type == "roberta":
                    param_i = base_param["out_proj.weight"]
                    param_j = graft_param["out_proj.weight"]
                elif self.model_type == "bert":
                    param_i = base_param["weight"]
                    param_j = graft_param["weight"]
                sigmoid = nn.Sigmoid()
                param_direction = torch.abs(param_i - param_j)
                param_direction = self.out_assess(param_direction)
                w_local = sigmoid(param_direction)

                param_uniting = param_i * w_local + param_j * (1 - w_local)

                if self.model_type == "roberta":
                    new_param["out_proj.weight"] = param_uniting
                elif self.model_type == "bert":
                    new_param["weight"] = param_uniting

                base_model.classifier.load_state_dict(new_param)
            return base_model

# v8b: global
class UnitingModelForNLP(torch.nn.Module):
        def __init__(self, config, model_type):
            super(UnitingModelForNLP, self).__init__()

            self.module = []

            self.out_assess = nn.Linear(config.hidden_size, config.hidden_size)
            initializer.default_weight_init(self.out_assess.weight)
            initializer.default_bias_init(self.out_assess.bias)
            self.module.append(self.out_assess)

            if model_type == "roberta":
                classifier_dropout = (
                    config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
                )
                self.dropout = nn.Dropout(classifier_dropout)

            self.model_type = model_type
            self.a = 0.4
            self.c = 500

        def forward(self, x, base_param, graft_param):
            if self.model_type == "roberta":
                base_param_weight = ["dense.weight", "out_proj.weight"]
                base_param_bias = ["dense.bias", "out_proj.bias"]
            elif self.model_type == "bert":
                base_param_weight = ["weight"]
                base_param_bias = ["bias"]

            def entropy(x, n=10):
                x = x.reshape(-1)
                scale = (x.max() - x.min()) / n
                entropy = torch.tensor(0.).to(x.device)
                for i in range(n):
                    p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale),
                                  dtype=torch.float) / len(x)
                    if p != 0:
                        entropy -= p * torch.log(p)
                return entropy


            if self.model_type == "roberta":
                x = x[:, 0, :]

                param_i = base_param["out_proj.weight"]
                param_j = graft_param["out_proj.weight"]
            elif self.model_type == "bert":
                param_i = base_param["weight"]
                param_j = graft_param["weight"]
            sigmoid = nn.Sigmoid()
            param_direction = torch.abs(param_i - param_j)
            param_direction = self.out_assess(param_direction)
            w_local = sigmoid(param_direction)
            # w_local = param_direction
            w_global = self.a / torch.pi * torch.arctan(self.c * (entropy(param_i) - entropy(param_j))) + 0.5

            param_uniting = param_i * w_global + param_j * (1 - w_global)

            if self.model_type == "roberta":
                x = self.dropout(x)
                x = torch.matmul(x, base_param["dense.weight"].transpose(0, 1)) + base_param["dense.bias"]
                x = torch.tanh(x)
                x = self.dropout(x)
                x = torch.matmul(x, param_uniting.transpose(0, 1)) + base_param["out_proj.bias"]
            elif self.model_type == "bert":
                x = torch.matmul(x, param_uniting.transpose(0, 1)) + base_param["bias"]

            return x

        def load_param(self, base_model, base_param, graft_param):
            new_param = copy.deepcopy(base_param)

            if self.model_type == "roberta":
                base_param_weight = ["dense.weight", "out_proj.weight"]
                base_param_bias = ["dense.bias", "out_proj.bias"]

            def entropy(x, n=10):
                x = x.reshape(-1)
                scale = (x.max() - x.min()) / n
                entropy = torch.tensor(0.).to(x.device)
                for i in range(n):
                    p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale),
                                  dtype=torch.float) / len(x)
                    if p != 0:
                        entropy -= p * torch.log(p)
                return entropy

            with torch.no_grad():
                if self.model_type == "roberta":
                    param_i = base_param["out_proj.weight"]
                    param_j = graft_param["out_proj.weight"]
                elif self.model_type == "bert":
                    param_i = base_param["weight"]
                    param_j = graft_param["weight"]
                sigmoid = nn.Sigmoid()
                param_direction = torch.abs(param_i - param_j)
                param_direction = self.out_assess(param_direction)
                w_local = sigmoid(param_direction)
                # w_local = param_direction
                w_global = self.a / torch.pi * torch.arctan(self.c * (entropy(param_i) - entropy(param_j))) + 0.5

                param_uniting = param_i * w_global + param_j * (1 - w_global)

                if self.model_type == "roberta":
                    new_param["out_proj.weight"] = param_uniting
                elif self.model_type == "bert":
                    new_param["weight"] = param_uniting

                base_model.classifier.load_state_dict(new_param)
            return base_model

# v6: v1上融合方式改为hard fusion
class UnitingModelForNLP_v6(torch.nn.Module):
    def __init__(self, config, model_type):
        super(UnitingModelForNLP, self).__init__()

        self.module = []

        self.dense_assess = nn.Linear(config.hidden_size, config.hidden_size)
        initializer.default_weight_init(self.dense_assess.weight)
        initializer.default_bias_init(self.dense_assess.bias)
        self.module.append(self.dense_assess)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.out_assess = nn.Linear(config.hidden_size, config.hidden_size)
        initializer.default_weight_init(self.out_assess.weight)
        initializer.default_bias_init(self.out_assess.bias)
        self.module.append(self.out_assess)

        self.model_type = model_type
        self.a = 0.4
        self.c = 500


    def forward(self, x, base_param, graft_param):
        if self.model_type == "roberta":
            base_param_weight = ["dense.weight", "out_proj.weight"]
            base_param_bias = ["dense.bias", "out_proj.bias"]

        def entropy(x, n=10):
            x = x.reshape(-1)
            scale = (x.max() - x.min()) / n
            entropy = torch.tensor(0.).to(x.device)
            for i in range(n):
                p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale), dtype=torch.float) / len(x)
                if p != 0:
                    entropy -= p * torch.log(p)
            return entropy

        x = x[:, 0, :]
        for i in range(len(base_param_weight)):
            param_i = base_param[base_param_weight[i]]
            param_j = graft_param[base_param_weight[i]]
            sigmoid = nn.Sigmoid()
            param_direction = torch.abs(param_i - param_j)
            param_direction = self.module[i](param_direction)
            w_local = sigmoid(param_direction)
            # w_local = param_direction
            w_global = self.a / torch.pi * torch.arctan(self.c * (entropy(param_i) - entropy(param_j))) + 0.5
            w_i = w_global * w_local + (1 - w_global)
            w_j = 1 - w_i

            # hard fusion
            param_uniting = torch.where(w_i > w_j, param_i, param_j)

            x = self.dropout(x)
            x = torch.matmul(x, param_uniting.transpose(0, 1)) + base_param[base_param_bias[i]]

            if i + 1 != len(base_param_weight):
                x = torch.tanh(x)
            else:
                return x

    def load_param(self, base_model, base_param, graft_param):
        new_param = copy.deepcopy(base_param)

        if self.model_type == "roberta":
            base_param_weight = ["dense.weight", "out_proj.weight"]
            base_param_bias = ["dense.bias", "out_proj.bias"]

        def entropy(x, n=10):
            x = x.reshape(-1)
            scale = (x.max() - x.min()) / n
            entropy = torch.tensor(0.).to(x.device)
            for i in range(n):
                p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale), dtype=torch.float) / len(x)
                if p != 0:
                    entropy -= p * torch.log(p)
            return entropy

        with torch.no_grad():
            # base_param = base_model.classifier.state_dict()
            for i in range(len(base_param_weight)):
                param_i = base_param[base_param_weight[i]]
                param_j = graft_param[base_param_weight[i]]

                sigmoid = nn.Sigmoid()
                param_direction = param_i - param_j
                param_direction = self.module[i](param_direction)
                w_local = sigmoid(param_direction)
                w_global = self.a / torch.pi * torch.arctan(self.c * (entropy(param_i) - entropy(param_j))) + 0.5

                w_i = w_global * w_local + (1 - w_global)
                w_j = 1 - w_i

                # hard fusion
                param_uniting = torch.where(w_i > w_j, param_i, param_j)

                # w_base = 1 - w_global * torch.exp(-w_global * w_local)
                # w_graft = 1 - w_global * torch.exp(w_global * w_local - 1)
                #
                # w = torch.stack([w_base, w_graft], dim=-1)
                # softmax = nn.Softmax(dim=-1)
                # w = softmax(w)
                # w_base, w_graft = torch.split(w, 1, dim=-1)
                # w_base = w_base.squeeze(dim=-1)
                # w_graft = w_graft.squeeze(dim=-1)
                #
                # param_uniting = param_i * w_base + param_j * w_graft

                new_param[base_param_weight[i]] = param_uniting

            base_model.classifier.load_state_dict(new_param)
        return base_model

# v7: v1上只用local
class UnitingModelForNLP_v7(torch.nn.Module):
    def __init__(self, config, model_type):
        super(UnitingModelForNLP, self).__init__()

        self.module = []

        self.dense_assess = nn.Linear(config.hidden_size, config.hidden_size)
        initializer.default_weight_init(self.dense_assess.weight)
        initializer.default_bias_init(self.dense_assess.bias)
        self.module.append(self.dense_assess)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.out_assess = nn.Linear(config.hidden_size, config.hidden_size)
        initializer.default_weight_init(self.out_assess.weight)
        initializer.default_bias_init(self.out_assess.bias)
        self.module.append(self.out_assess)

        self.model_type = model_type
        self.a = 0.4
        self.c = 500


    def forward(self, x, base_param, graft_param):
        if self.model_type == "roberta":
            base_param_weight = ["dense.weight", "out_proj.weight"]
            base_param_bias = ["dense.bias", "out_proj.bias"]

        def entropy(x, n=10):
            x = x.reshape(-1)
            scale = (x.max() - x.min()) / n
            entropy = torch.tensor(0.).to(x.device)
            for i in range(n):
                p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale), dtype=torch.float) / len(x)
                if p != 0:
                    entropy -= p * torch.log(p)
            return entropy

        x = x[:, 0, :]
        for i in range(len(base_param_weight)):
            param_i = base_param[base_param_weight[i]]
            param_j = graft_param[base_param_weight[i]]
            sigmoid = nn.Sigmoid()
            param_direction = torch.abs(param_i - param_j)
            param_direction = self.module[i](param_direction)
            w_local = sigmoid(param_direction)

            param_uniting = param_i * w_local + param_j * (1 - w_local)

            x = self.dropout(x)
            x = torch.matmul(x, param_uniting.transpose(0, 1)) + base_param[base_param_bias[i]]

            if i + 1 != len(base_param_weight):
                x = torch.tanh(x)
            else:
                return x

    def load_param(self, base_model, base_param, graft_param):
        new_param = copy.deepcopy(base_param)

        if self.model_type == "roberta":
            base_param_weight = ["dense.weight", "out_proj.weight"]
            base_param_bias = ["dense.bias", "out_proj.bias"]

        def entropy(x, n=10):
            x = x.reshape(-1)
            scale = (x.max() - x.min()) / n
            entropy = torch.tensor(0.).to(x.device)
            for i in range(n):
                p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale), dtype=torch.float) / len(x)
                if p != 0:
                    entropy -= p * torch.log(p)
            return entropy

        with torch.no_grad():
            # base_param = base_model.classifier.state_dict()
            for i in range(len(base_param_weight)):
                param_i = base_param[base_param_weight[i]]
                param_j = graft_param[base_param_weight[i]]

                sigmoid = nn.Sigmoid()
                param_direction = param_i - param_j
                param_direction = self.module[i](param_direction)
                w_local = sigmoid(param_direction)

                param_uniting = param_i * w_local + param_j * (1 - w_local)

                # w_base = 1 - w_global * torch.exp(-w_global * w_local)
                # w_graft = 1 - w_global * torch.exp(w_global * w_local - 1)
                #
                # w = torch.stack([w_base, w_graft], dim=-1)
                # softmax = nn.Softmax(dim=-1)
                # w = softmax(w)
                # w_base, w_graft = torch.split(w, 1, dim=-1)
                # w_base = w_base.squeeze(dim=-1)
                # w_graft = w_graft.squeeze(dim=-1)
                #
                # param_uniting = param_i * w_base + param_j * w_graft

                new_param[base_param_weight[i]] = param_uniting

            base_model.classifier.load_state_dict(new_param)
        return base_model

# v8: v1上只用global
class UnitingModelForNLP_v8(torch.nn.Module):
    def __init__(self, config, model_type):
        super(UnitingModelForNLP, self).__init__()

        self.module = []

        self.dense_assess = nn.Linear(config.hidden_size, config.hidden_size)
        initializer.default_weight_init(self.dense_assess.weight)
        initializer.default_bias_init(self.dense_assess.bias)
        self.module.append(self.dense_assess)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.out_assess = nn.Linear(config.hidden_size, config.hidden_size)
        initializer.default_weight_init(self.out_assess.weight)
        initializer.default_bias_init(self.out_assess.bias)
        self.module.append(self.out_assess)

        self.model_type = model_type
        self.a = 0.4
        self.c = 500


    def forward(self, x, base_param, graft_param):
        if self.model_type == "roberta":
            base_param_weight = ["dense.weight", "out_proj.weight"]
            base_param_bias = ["dense.bias", "out_proj.bias"]

        def entropy(x, n=10):
            x = x.reshape(-1)
            scale = (x.max() - x.min()) / n
            entropy = torch.tensor(0.).to(x.device)
            for i in range(n):
                p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale), dtype=torch.float) / len(x)
                if p != 0:
                    entropy -= p * torch.log(p)
            return entropy

        x = x[:, 0, :]
        for i in range(len(base_param_weight)):
            param_i = base_param[base_param_weight[i]]
            param_j = graft_param[base_param_weight[i]]

            w_global = self.a / torch.pi * torch.arctan(self.c * (entropy(param_i) - entropy(param_j))) + 0.5

            param_uniting = param_i * w_global + param_j * (1 - w_global)

            x = self.dropout(x)
            x = torch.matmul(x, param_uniting.transpose(0, 1)) + base_param[base_param_bias[i]]

            if i + 1 != len(base_param_weight):
                x = torch.tanh(x)
            else:
                return x

    def load_param(self, base_model, base_param, graft_param):
        new_param = copy.deepcopy(base_param)

        if self.model_type == "roberta":
            base_param_weight = ["dense.weight", "out_proj.weight"]
            base_param_bias = ["dense.bias", "out_proj.bias"]

        def entropy(x, n=10):
            x = x.reshape(-1)
            scale = (x.max() - x.min()) / n
            entropy = torch.tensor(0.).to(x.device)
            for i in range(n):
                p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale), dtype=torch.float) / len(x)
                if p != 0:
                    entropy -= p * torch.log(p)
            return entropy

        with torch.no_grad():
            # base_param = base_model.classifier.state_dict()
            for i in range(len(base_param_weight)):
                param_i = base_param[base_param_weight[i]]
                param_j = graft_param[base_param_weight[i]]

                sigmoid = nn.Sigmoid()
                param_direction = param_i - param_j
                param_direction = self.module[i](param_direction)
                w_local = sigmoid(param_direction)
                w_global = self.a / torch.pi * torch.arctan(self.c * (entropy(param_i) - entropy(param_j))) + 0.5

                param_uniting = param_i * w_global + param_j * (1 - w_global)

                # w_base = 1 - w_global * torch.exp(-w_global * w_local)
                # w_graft = 1 - w_global * torch.exp(w_global * w_local - 1)
                #
                # w = torch.stack([w_base, w_graft], dim=-1)
                # softmax = nn.Softmax(dim=-1)
                # w = softmax(w)
                # w_base, w_graft = torch.split(w, 1, dim=-1)
                # w_base = w_base.squeeze(dim=-1)
                # w_graft = w_graft.squeeze(dim=-1)
                #
                # param_uniting = param_i * w_base + param_j * w_graft

                new_param[base_param_weight[i]] = param_uniting

            base_model.classifier.load_state_dict(new_param)
        return base_model

# pruning v1:0.02 for roberta
class PruningModelForNLP_ro(torch.nn.Module):
    def __init__(self, config, model_type):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)

        dense_weight_mask = self.dense.weight.data > 0.02
        dense_bias_mask = self.dense.bias.data > 0.02
        self.dense.weight.data *= dense_weight_mask.float()
        self.dense.bias.data *= dense_bias_mask.float()

        out_proj_weight_mask = self.out_proj.weight.data > 0.02
        out_proj_bias_mask = self.out_proj.bias.data > 0.02
        self.out_proj.weight.data *= out_proj_weight_mask.float()
        self.out_proj.bias.data *= out_proj_bias_mask.float()

        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

# pruning v1:0.02 for bert
class PruningModelForNLP(torch.nn.Module):
    def __init__(self, config, model_type):
        super().__init__()

        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, model):

        out_proj_weight_mask = model.classifier.weight.data > 0.02
        out_proj_bias_mask = model.classifier.bias.data > 0.02
        model.classifier.weight.data *= out_proj_weight_mask.float()
        model.classifier.bias.data *= out_proj_bias_mask.float()

        return model
    
# # pruning v2:0.2 for roberta
# class PruningModelForNLP_v2_ro(torch.nn.Module):
#     def __init__(self, config, model_type):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         classifier_dropout = (
#             config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
#         )
#         self.dropout = nn.Dropout(classifier_dropout)
#         self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
#
#     def forward(self, features, **kwargs):
#         x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
#         x = self.dropout(x)
#
#         dense_weight_mask = self.dense.weight.data > 0.2
#         dense_bias_mask = self.dense.bias.data > 0.2
#         self.dense.weight.data *= dense_weight_mask.float()
#         self.dense.bias.data *= dense_bias_mask.float()
#
#         out_proj_weight_mask = self.out_proj.weight.data > 0.2
#         out_proj_bias_mask = self.out_proj.bias.data > 0.2
#         self.out_proj.weight.data *= out_proj_weight_mask.float()
#         self.out_proj.bias.data *= out_proj_bias_mask.float()
#
#         x = self.dense(x)
#         x = torch.tanh(x)
#         x = self.dropout(x)
#         x = self.out_proj(x)
#         return x

# for roberta
class EnsembleModelForNLP_ro(torch.nn.Module):
    def __init__(self, model_i, model_j):
        super().__init__()
        self.classifier_i = model_i.classifier
        self.classifier_j = model_j.classifier

    def forward(self, sequence_output):
        logit_i = self.classifier_i(sequence_output)
        logit_j = self.classifier_j(sequence_output)

        return (logit_i + logit_j) / 2

# for bert
class EnsembleodelForNLP(torch.nn.Module):
    def __init__(self, model_i, model_j):
        super().__init__()
        self.classifier_i = model_i.classifier
        self.classifier_j = model_j.classifier

    def forward(self, sequence_output):
        logit_i = self.classifier_i(sequence_output)
        logit_j = self.classifier_j(sequence_output)

        return (logit_i + logit_j) / 2