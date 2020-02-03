from copy import deepcopy
from typing import Union, Optional
from torch import nn, Tensor
from .module import init_module
from .typing import io_type


# https://stackoverflow.com/questions/49888155/class-cannot-subclass-qobject-has-type-any-using-mypy  # noqa: E501
class JSONNet(nn.Module):  # type: ignore
    def _add_modules(self, module_dict: dict) -> None:
        # sort the attr's name, as this order can be random.
        # across different Python runs.
        # so sort it to get stable initialization order.
        for attrname in sorted(module_dict.keys()):
            modspec = module_dict[attrname]
            self.moduledict[attrname] = init_module(modspec['name'],
                                                    modspec['params'],
                                                    modspec['init'])

    def get_module(self, name: str) -> nn.Module:
        return self.moduledict[name]

    def get_module_optional(self, name: str) -> Optional[nn.Module]:
        return self.get_module(name) if name in self.moduledict else None

    def get_param_dict(self):
        # so that I can inspect inside. Sometimes I need to
        # put some extra info in param dict to facilitate

        return deepcopy(self.__param_dict)

    def __init__(self, param_dict: dict) -> None:
        super().__init__()
        self.__param_dict = param_dict
        self.moduledict = nn.ModuleDict()

        self._add_modules(param_dict['module_dict'])

        # this can be used to put useful aux info.
        # for example, for maskcnn type models,
        # you can use this to provide references to
        # conv layers that need regularization, instead of writing slightly
        # different loss function implementations every time.
        # (for maskcnn type networks with different conv implementations)

        self.extradict = dict()

    def forward(self, *inputs: list, state_dict: Union[dict, None] = None,
                verbose: bool = False) -> io_type:
        # state_dict is typically used for debugging,
        # checking internal state, etc.
        # it should be a subset of intermediate_dict
        # I will basically do things according to op_list
        temp_dict: dict = {'inputs': tuple(inputs)}
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
    def get_io(io_spec: Optional[Union[str, list]],
               temp_dict: dict) -> io_type:
        if isinstance(io_spec, str):
            ret = temp_dict[io_spec]
        elif isinstance(io_spec, list):
            ret = tuple(temp_dict[x] for x in io_spec)
        elif io_spec is None:
            raise RuntimeError
            # really, if something does not take IO, then the computation should not be
            # done at runtime.
            # return None  # some op takes no input.
        else:
            raise NotImplementedError

        JSONNet.check_io(ret)
        return ret

    @staticmethod
    def set_io(name: str, value: io_type, temp_dict: dict) -> None:
        # check it's indeed io_type
        # https://stackoverflow.com/questions/28100078/calling-static-method-from-inside-the-class
        JSONNet.check_io(value)
        temp_dict[name] = value

    @staticmethod
    def check_io(value: io_type) -> None:
        # check it's indeed io_type
        if not isinstance(value, Tensor):
            assert isinstance(value, tuple)
            for _ in value:
                assert isinstance(_, Tensor)

    def _forward_one_op(self, op_spec_full: dict, temp_dict: dict) -> None:
        from .op import get_op
        # different kinds of ops.
        #
        # function op, concat, sequential, split, indexing, etc. they can
        # take additional parameters apart from op_in. also, we have detach.
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
        assert op_spec_full.keys() == {'in', 'out', 'name', 'args', 'kwargs'}
        op_this = get_op(self, {k: op_spec_full[k] for k in
                                {'name', 'args', 'kwargs'}})
        in_this = self.get_io(op_spec_full['in'], temp_dict)
        self.set_io(op_spec_full['out'], op_this(in_this), temp_dict)
