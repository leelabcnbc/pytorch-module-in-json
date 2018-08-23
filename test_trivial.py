"""a trivial demo, with one Linear layer"""
from torch import tensor
from torchnetjson.builder import build_net
module_dict = {
    'fc': ['torch.nn.linear', {'in_features': 3, 'out_features': 5}]
}

op_list = [
    {'name': 'module',
     'args': ['fc'],
     'kwargs': {},
     'in': 'input0',
     'out': 'out'
     }
]

param_dict = {
    'module_dict': module_dict,
    'op_list': op_list,
    'out': 'out'
}

net = build_net(param_dict)

input_this = tensor([1.0, 2.0, 3.0])
output_this = net(input_this, verbose=True).detach()

print(output_this, type(output_this))
