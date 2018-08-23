from typing import Union
from torch import nn
from .module import init_module
from .typing import io_type


# https://stackoverflow.com/questions/49888155/class-cannot-subclass-qobject-has-type-any-using-mypy  # noqa: E501
class JSONNet(nn.Module):  # type: ignore
    def _add_modules(self, module_dict: dict) -> None:
        for attrname, modspec in module_dict.items():
            self.add_module(attrname, init_module(modspec['name'],
                                                  modspec['params'],
                                                  modspec['init']))

    def get_module(self, name: str) -> nn.Module:
        return self._modules[name]

    def __init__(self, param_dict: dict) -> None:
        super().__init__()
        self.__param_dict = param_dict
        self._add_modules(param_dict['module_dict'])
        # TODO some initialization stuff later.
        # or I can do initialization in the module_dict directly.
        # but that maybe a bit constrained.
        # if you want to do some initialization that depends on multiple
        # modules.
        # I like to separate this from architecture spec.

    def forward(self, *inputs: list, state_dict: Union[dict, None] = None,
                verbose: bool = False) -> io_type:
        # state_dict is typically used for debugging,
        # checking internal state, etc.
        # it should be a subset of intermediate_dict
        # I will basically do things according to op_list
        temp_dict: dict = {'inputs': inputs}
        temp_dict.update({f'input{i}': v for i, v in enumerate(inputs)})
        if verbose:
            print('=====initial temp dict=====')
            print(temp_dict)
            print('\n')
        op_list = self.__param_dict['op_list']
        for i, op_spec in enumerate(op_list):
            if verbose:
                print(f'=====op {i} start=====')
            # this function does the actual work.
            # it can be recursively called.
            self._forward_one_op(op_spec, temp_dict)
            if verbose:
                print(f'=====op {i} end=====')
                print('\n')
        if verbose:
            print('=====final temp dict=====')
            print(temp_dict)
            print('\n')

        return self.get_io(self.__param_dict['out'], temp_dict)

    @staticmethod
    def get_io(io_spec: Union[str, list], temp_dict: dict) -> io_type:
        if isinstance(io_spec, str):
            return temp_dict[io_spec]
        elif isinstance(io_spec, list):
            return [temp_dict[x] for x in io_spec]
        else:
            raise NotImplementedError

    def _forward_one_op(self, op_spec: dict, temp_dict: dict) -> None:
        from .op import get_op
        # different kinds of ops.
        #
        # function op, concat, sequential, split, indexing, etc. they can take
        #   additional parameters apart from op_in. also, we have detach.
        #   for saving space
        # module op, using defined modules. all they take is op_in
        #
        # op_name, op_params, op_in, op_out =
        # ins can be a list, which contains a fixed number of inputs
        # and it will be unpacked/packed when passed into ops,
        # depending what you want.
        #
        # each op has its own packing/unpacking semantics
        # and you can override when the input is a list when applicable,
        # by passing additional parameters into the op.
        #
        # for example, torch.cat is packing.
        #
        # for sequential, I favor unpacking,
        # which matches forward()'s API more closely.
        # (which is violated by torch's own Sequential, which assumes
        # single input).
        #
        # or a string, which is just a single input
        #
        # out will always be taken unchanged,
        # which can be a list/tuple or singleton.
        # splitting it is at the user's own disposal for now.
        op_this = get_op(self, op_spec['name'],
                         op_spec['args'], op_spec['kwargs'])
        temp_dict[op_spec['out']] = op_this(self.get_io(op_spec['in'],
                                                        temp_dict))
