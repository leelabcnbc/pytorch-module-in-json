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
from typing import Dict
import torch
from torch.nn import functional
from .net import JSONNet
from .typing import io_type, op_constructor_type, op_type


def get_op(net: JSONNet, op_spec: dict) -> op_type:
    # TODO: do such check before doing forward()
    assert op_spec.keys() == {'name', 'args', 'kwargs'}
    return _get_op_inner(net, op_spec['name'],
                         op_spec['args'], op_spec['kwargs'])


def _get_op_inner(net: JSONNet, op_name: str,
                  args: list, kwargs: dict) -> op_type:
    """

    :param net: the network to be built.
    :param op_name: name of the operation.
    :param args: positional args of the operation.
    :param kwargs: kwargs args of the operation.
    :return: a function that maps input to output.
    """
    op_constructor = _op_dict[op_name]
    return op_constructor(net, *args, **kwargs)


def _module_op(net: JSONNet, module_name: str, *,
               unpack: bool = True) -> op_type:
    # get module
    def _op_fn(inputs: io_type) -> io_type:
        mod = net.get_module(module_name)
        if isinstance(inputs, torch.Tensor) or not unpack:
            return mod(inputs)
        else:
            return mod(*inputs)

    return _op_fn


def _module_repeat_op(net: JSONNet, module_name: str):
    # this supports applying a module to each element of an input.
    # `inputs` can be only tuple of Tensor,
    # module here only supports ONE Tensor input and return ONE Tensor output
    def _op_fn(inputs: io_type) -> io_type:
        mod = net.get_module(module_name)
        ret = []
        assert isinstance(inputs, tuple)
        for x in inputs:
            ret.append(mod(x))
        return tuple(ret)

    return _op_fn


def _sequential_op(net: JSONNet, list_of_op_specs: list) -> op_type:
    # get module list
    op_list = [get_op(net, op_spec) for op_spec in list_of_op_specs]

    def _op_fn(inputs: io_type) -> io_type:
        for op in op_list:
            inputs = op(inputs)
        return inputs

    return _op_fn


def _detach_op(net: JSONNet) -> op_type:
    def _op_fn(inputs: io_type) -> io_type:
        if isinstance(inputs, torch.Tensor):
            return inputs.detach()
        elif isinstance(inputs, tuple):
            return tuple(x.detach() for x in inputs)
        else:
            raise TypeError

    return _op_fn


def _sum_op(net: JSONNet) -> op_type:
    def _op_fn(inputs: io_type) -> io_type:
        if isinstance(inputs, tuple):
            return sum(inputs)
        else:
            raise TypeError

    return _op_fn


def _stack_op(net: JSONNet, dim: int = 0) -> op_type:
    def _op_fn(inputs: io_type) -> io_type:
        if isinstance(inputs, tuple):
            return torch.stack(inputs, dim)
        else:
            raise TypeError

    return _op_fn


def _loss_op(net: JSONNet, *, loss_type: str, kwargs: dict,
             factor: float = 1.0) -> op_type:
    def _op_fn(inputs: io_type) -> io_type:
        if isinstance(inputs, tuple):
            computed, target = inputs
        elif isinstance(inputs, torch.Tensor):
            # single input, assume 0 as target.
            computed, target = inputs, torch.zeros_like(inputs)
        else:
            raise TypeError

        if loss_type == 'l1':
            loss_fn = functional.l1_loss
        elif loss_type == 'l2':
            loss_fn = functional.mse_loss
        else:
            raise ValueError

        return factor * loss_fn(computed, target, **kwargs)

    return _op_fn


_op_dict: Dict[str, op_constructor_type] = {
    'module': _module_op,
    'sequential': _sequential_op,
    'detach': _detach_op,
    'sum': _sum_op,
    'loss': _loss_op,
    'stack': _stack_op,
    'module_repeat': _module_repeat_op,
}


def register_op_custom(name: str, op: op_constructor_type) -> None:
    # this should be called by user code to register modules they want.
    # TODO: more strict check later.
    assert name.lower() == name
    assert name.startswith('custom.')
    _op_dict[name] = op
