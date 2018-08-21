import json

from torch import nn

from .module import init_module


class _JSONNet(nn.Module):
    def _add_modules(self, module_dict):
        for attrname, (modname, modparams) in module_dict.items():
            self.add_module(attrname, init_module(modname, modparams))

    def get_module(self, name):
        return self._modules[name]

    def __init__(self, param_dict):
        super().__init__()
        self.__param_dict = param_dict
        self._add_modules(param_dict['module_dict'])
        # TODO some initialization stuff later.
        # I like to separate this from architecture spec.

    def forward(self, *inputs, state_dict=None):
        # state_dict is typically used for debugging,
        # checking internal state, etc.
        # it should be a subset of intermediate_dict
        # I will basically do things according to op_list
        intermediate_dict = {'inputs': inputs}
        op_list = self.__param_dict['op_list']
        for op_spec in op_list:
            # op_name, op_params, op_in, op_out =
            # ins can be a list, which contains a fixed number of inputs
            # and it will be unpacked when passed into module ops,
            # and depends with passed into other ops.
            # these should be customized later.
            # or a string, which is just a single input
            # out will always be taken unchanged,
            # which can be a list or singleton.
            # splitting it is at the user's own disposal for now.
            pass


def build_net(param_dict):
    # make sure it's really JSON serializable.
    param_dict_normalized = json.loads(json.dumps(param_dict))
    assert param_dict_normalized == param_dict
    return _JSONNet(param_dict_normalized)
