"""a trivial demo, with one Linear layer"""
from torch import tensor
from torchnetjson.builder import build_net
module_dict = {
    'fc1': {'name':'torch.nn.linear',
           'params': {'in_features': 3, 'out_features': 5},
           'init': None},
    'fc2': {'name':'torch.nn.linear',
           'params': {'in_features': 5, 'out_features': 3},
           'init': None}
}

op_spec_list = [
    {'name': 'module',
     'args': ['fc1'],
     'kwargs': {},
     },
    {'name': 'module',
     'args': ['fc2'],
     'kwargs': {},
     },
]

op_list = [
    {
        'name': 'sequential',
        'args': [op_spec_list,],
        'kwargs': {},
        'in': 'input0',
        'out': 'out',
    },
    {
        'name': 'detach',
        'args': [],
        'kwargs': {},
        'in': 'out',
        'out': 'out_detached',
    }
]

param_dict = {
    'module_dict': module_dict,
    'op_list': op_list,
    'out': 'out_detached'
}

net = build_net(param_dict)

input_this = tensor([1.0, 2.0, 3.0])
output_this = net(input_this, verbose=True).detach()

print(output_this, type(output_this))
