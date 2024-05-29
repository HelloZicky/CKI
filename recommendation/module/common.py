"""
Common modules
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from . import initializer


class StackedDense(torch.nn.Module):
    def __init__(self, in_dimension, units, activation_fns):
        super(StackedDense, self).__init__()

        modules = []
        units = [in_dimension] + list(units)
        for i in range(1, len(units)):
            linear = torch.nn.Linear(units[i - 1], units[i], bias=True)
            initializer.default_weight_init(linear.weight)
            initializer.default_bias_init(linear.bias)
            modules.append(linear)

            if activation_fns[i - 1] is not None:
                modules.append(activation_fns[i - 1]())

        self.net = torch.nn.Sequential(*modules)

    def __setitem__(self, k, v):
        self.k = v

    def forward(self, x):
        return self.net(x)


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

class Graft_StackedDense(torch.nn.Module):
    def __init__(self, in_dimension, units, activation_fns, model_name):
        super(Graft_StackedDense, self).__init__()

        self._net_modules = []
        self._param_net = []
        self._activation_modules = []
        self.model_name = model_name
        units = [in_dimension] + list(units)
        self.units = units

        self.a = 0.4
        self.c = 500

        for i in range(1, len(units)):
            linear = torch.nn.Linear(units[i - 1], units[i], bias=True)
            initializer.default_weight_init(linear.weight)
            initializer.default_bias_init(linear.bias)
            self.add_module(f'_net_module_{i - 1}', linear)
            self._net_modules.append(linear)

            if activation_fns[i - 1] is not None:
                self._activation_modules.append(activation_fns[i - 1]())

            param_net_module = Linear(
                units[i - 1],
                units[i - 1],
                bias=False
            )

            self.add_module(f'_param_net_{i - 1}', param_net_module)
            self._param_net.append(param_net_module)

    def forward(self, x, graft_param, finetune=False):
        units = self.units
        if self.model_name == "graft_din":
            base_param_name = ["_classifier.net.0.weight", "_classifier.net.2.weight", "_classifier.net.4.weight"]
        elif self.model_name == "graft_gru4rec" or self.model_name == "graft_sasrec":
            base_param_name = ["_classifier.net.0.weight", "_classifier.net.2.weight"]

        def entropy(x, n=10):
            x = x.reshape(-1)
            scale = (x.max() - x.min()) / n
            entropy = torch.tensor(0.).to(x.device)
            for i in range(n):
                p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale), dtype=torch.float) / len(x)
                if p != 0:
                    entropy -= p * torch.log(p)
            return entropy

        if finetune:
            for i in range(1, len(units)):
                index = i - 1

                x = self._net_modules[index](x)
                x = self._activation_modules[index](x) if index < len(self._activation_modules) else x
            return x
        else:
            for i in range(1, len(units)):
                index = i - 1
                base_param = self._net_modules[index].state_dict()

                sigmoid = nn.Sigmoid()
                graft_weight = graft_param[base_param_name[index]]
                base_weight = base_param["weight"]
                param_direction = torch.abs(base_weight - graft_weight)
                param_direction = self._param_net[index](param_direction)
                w_local = sigmoid(param_direction)
                w_global = self.a / torch.pi * torch.arctan(self.c * (entropy(base_weight) - entropy(graft_weight))) + 0.5

                w_base = w_global * (1 - torch.exp(-w_global * w_local))
                w_graft = (1 - w_global) * (1 - torch.exp(-(1 - w_global) * (1 - w_local)))

                w = torch.stack([w_base, w_graft], dim=-1)
                softmax = nn.Softmax(dim=-1)
                w = softmax(w)
                w_base, w_graft = torch.split(w, 1, dim=-1)
                w_base = w_base.squeeze(dim=-1)
                w_graft = w_graft.squeeze(dim=-1)

                base_param["weight"] = base_weight * w_base + graft_weight * w_graft
                # self._net_modules[index].load_state_dict(base_param)
                x = torch.matmul(x, base_param["weight"].transpose(0, 1)) + base_param["bias"]
                x = self._activation_modules[index](x) if index < len(self._activation_modules) else x

            return x

    def forward_test(self, x):
        units = self.units

        for i in range(1, len(units)):
            index = i - 1

            x = self._net_modules[index](x)
            x = self._activation_modules[index](x) if index < len(self._activation_modules) else x
        return x

    def load_param(self, graft_param):
        units = self.units
        if self.model_name == "graft_din":
            base_param_name = ["_classifier.net.0.weight", "_classifier.net.2.weight", "_classifier.net.4.weight"]
        elif self.model_name == "graft_gru4rec" or self.model_name == "graft_sasrec":
            base_param_name = ["_classifier.net.0.weight", "_classifier.net.2.weight"]

        def entropy(x, n=10):
            x = x.reshape(-1)
            scale = (x.max() - x.min()) / n
            entropy = torch.tensor(0.).to(x.device)
            for i in range(n):
                p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale), dtype=torch.float) / len(x)
                if p != 0:
                    entropy -= p * torch.log(p)
            return entropy

        for i in range(1, len(units)):
            index = i - 1
            base_param = self._net_modules[index].state_dict()

            sigmoid = nn.Sigmoid()
            graft_weight = graft_param[base_param_name[index]]
            base_weight = base_param["weight"]
            param_direction = base_weight - graft_weight
            param_direction = self._param_net[index](param_direction)
            w_local = sigmoid(param_direction)
            w_global = self.a / torch.pi * torch.arctan(self.c * (entropy(base_weight) - entropy(graft_weight))) + 0.5

            w_base = 1 - w_global * torch.exp(-w_global * w_local)
            w_graft = 1 - w_global * torch.exp(w_global * w_local - 1)

            w = torch.stack([w_base, w_graft], dim=-1)
            softmax = nn.Softmax(dim=-1)
            w = softmax(w)
            w_base, w_graft = torch.split(w, 1, dim=-1)
            w_base = w_base.squeeze(dim=-1)
            w_graft = w_graft.squeeze(dim=-1)

            base_param["weight"] = base_weight * w_base + graft_weight * w_graft
            self._net_modules[index].load_state_dict(base_param)


class HyperNetwork_FC(nn.Module):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False):
        super(HyperNetwork_FC, self).__init__()

        self.modules = []
        units = [in_dimension] + list(units)
        self.activation_fns = activation_fns
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)
        self.expand = expand
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                self.modules.append(activation_fns[i - 1]())

    def forward(self, x, z, sample_num=32, trigger_seq_length=30):
        units = self.units
        duet_param = dict()
        user_state, _ = self._gru_cell(z)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        z = self._mlp_trans(user_state)

        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            weight = torch.matmul(z, self.w1[index]) + self.b1[index]

            output_size = units[i]

            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)
            bias = torch.matmul(z, self.w2[index]) + self.b2[index]
            duet_param["weight" + str(i)] = weight
            duet_param["bias" + str(i)] = bias
            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
            x = self.modules[index](x) if index < len(self.modules) else x

        return x, duet_param

class HyperNetwork_FC_mask(nn.Module):
    def __init__(self, in_dimension, units, activation_fns, base_param, model_name, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False):
        super(HyperNetwork_FC_mask, self).__init__()
        self.base_param = base_param
        self.model_name = model_name
        self.modules = []
        units = [in_dimension] + list(units)
        self.activation_fns = activation_fns
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)
        self.expand = expand

        self.a = 0.4
        self.c = 500
        self._param_net = []

        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                self.modules.append(activation_fns[i - 1]())

            param_net_module = Linear(
                output_size,
                output_size,
                bias=False
            )
            self.add_module(f'_param_net_{i - 1}', param_net_module)
            self._param_net.append(param_net_module)

    def forward(self, x, z, duet_param, sample_num=32, trigger_seq_length=30):
        units = self.units

        user_state, _ = self._gru_cell(z)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        base_param_name = list()
        # base_bias_name = list()

        z = self._mlp_trans(user_state)
        if self.model_name == "din":
            base_param_name = ["_classifier.net.0.weight", "_classifier.net.2.weight", "_classifier.net.4.weight"]
            # base_bias_name = ["_classifier.net.0.bias", "_classifier.net.2.bias", "_classifier.net.4.bias"]
        elif self.model_name == "gru4rec" or self.model_name == "sasrec":
            base_param_name = ["_classifier.net.0.weight", "_classifier.net.2.weight"]
            # base_bias_name = ["_classifier.net.0.bias", "_classifier.net.2.bias"]

        # v9
        def entropy(x, n=10):
            x = x.reshape(x.size()[0], -1)
            scale = (x.max(dim=1).values - x.min(dim=1).values) / n
            entropy = torch.zeros(x.size()[0]).to(x.device)

            for i in range(n):
                lower_bound = x.min(dim=1).values + i * scale
                upper_bound = x.min(dim=1).values + (i + 1) * scale

                count = torch.sum((x >= lower_bound.unsqueeze(1)) & (x < upper_bound.unsqueeze(1)), dim=1, dtype=torch.float)
                p = count / x.size()[1]

                non_zero_indices = p != 0
                entropy[non_zero_indices] -= p[non_zero_indices] * torch.log(p[non_zero_indices])

            return entropy

        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]
            base_weight = self.base_param[base_param_name[index]]
            duet_weight = duet_param["weight" + str(i)]
            param_value = torch.matmul(z, self.w1[index]) + self.b1[index]
            output_size = units[i]

            if not self.batch:
                param_value = param_value.view(input_size, output_size)

            else:
                param_value = param_value.view(sample_num, input_size, output_size)

            base_weight = base_weight.repeat(param_value.size()[0], 1, 1).permute(0, 2, 1)

            sigmoid = nn.Sigmoid()
            param_direction = torch.abs(duet_weight - base_weight)
            param_direction = self._param_net[index](param_direction)
            w_local = sigmoid(param_direction)

            # decimal_places = 2
            # w = torch.round((self.a / torch.pi * torch.arctan(self.c * (entropy(duet_weight) - entropy(base_weight))) + 0.5) * (10 ** decimal_places)) / (10**decimal_places)
            w_global = self.a / torch.pi * torch.arctan(self.c * (entropy(duet_weight) - entropy(base_weight))) + 0.5
            w_global = w_global.view(-1, 1, 1)

            w_duet = w_global * (1 - torch.exp(-w_global * w_local))
            w_base = (1 - w_global) * (1 - torch.exp(-(1 - w_global) * (1 - w_local)))
            w = torch.stack([w_duet, w_base], dim=-1)
            softmax = nn.Softmax(dim=-1)
            w = softmax(w)
            w_duet, w_base = torch.split(w, 1, dim=-1)
            w_duet = w_duet.squeeze(dim=-1)
            w_base = w_base.squeeze(dim=-1)

            mask = duet_weight * w_duet + base_weight * w_base

            bias = torch.matmul(z, self.w2[index]) + self.b2[index]

            x = torch.bmm(x.unsqueeze(1), mask).squeeze(1) + bias
            x = self.modules[index](x) if index < len(self.modules) else x

        return x


