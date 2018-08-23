# sequential (a linear chain of modules)
#    to support multi-input,
# we support both packing and unpacking style for intermediates
#
# torch's own sequential is packing style.
#
# unpacking style may be useful in some cases as well.
#
# concat
# index
# loop (which contains sub op list)
# singleton (a single module)
#
# everything should have a shared namespace for simplicity,
# i.e. for loop op, we should make sure it's intermediates has no
# name collision with outer intermediate.
# to achieve this, we should pass the same intermediate over and over
import torch
from .net import JSONNet


def get_op(net: JSONNet, op_name: str,
           args: list, kwargs: dict) -> function:
    op_this = _op_dict[op_name](net, *args, **kwargs)
    return op_this


def _module_op(net: JSONNet, module_name: str, *, unpack=True) -> function:
    # get module
    def module_op_fn(inputs):
        mod = net.get_module(module_name)
        if isinstance(inputs, torch.Tensor) or not unpack:
            return mod(inputs)
        else:
            return mod(*inputs)

    return module_op_fn


_op_dict = {
    'module': _module_op,
}
