"""initializers"""

from typing import Callable, Union, Iterable, Optional

from torch import nn

_init_mapping_official = dict()


def standard_init(mod: nn.Module, init: dict, *,
                  attrs_to_init: Optional[Iterable[str]]=None):
    if attrs_to_init is None:
        attrs_to_init = {'weight'}
    # works for those modules with `weight` and possibly `bias`.
    assert init.keys() == {'strategy', 'parameters'}

    init_inner = {
        'normal': nn.init.normal_,
        'constant': nn.init.constant_,
        'kaiming_normal': nn.init.kaiming_normal_
    }[init['strategy']]

    for attr in attrs_to_init:
        init_inner(getattr(mod, attr), **init['parameters'])

    if mod.bias is not None:
        nn.init.constant_(mod.bias, 0)


def bn_init_passthrough(mod: Union[nn.BatchNorm1d,
                                   nn.BatchNorm2d,
                                   nn.BatchNorm3d],
                        init: dict):
    # set scale to 1, and bias to 0
    assert init == {}
    if mod.affine:
        mod.bias.data.zero_()
        mod.weight.data.fill_(1)


def _register_init_official(name: str,
                            init: Callable[[nn.Module, dict], None]) -> None:
    # TODO: more strict check later.
    assert name.lower() == name
    assert name.startswith('torch.')
    _init_mapping_official[name] = init


def _register_init_official_loader() -> None:
    # TODO: make this kind of automatic.
    _register_init_official('torch.nn.conv2d', standard_init)
    _register_init_official('torch.nn.convtranspose2d', standard_init)
    _register_init_official('torch.nn.linear', standard_init)
    _register_init_official('torch.nn.batchnorm2d', bn_init_passthrough)


_register_init_official_loader()

_init_mapping_custom = dict()


def register_init_custom(name: str,
                         init: Callable[[nn.Module, dict], None]) -> None:
    # TODO: more strict check later.
    assert name.lower() == name
    assert not name.startswith('torch.')
    _init_mapping_custom[name] = init


def intialize_a_module(name: str, mod: nn.Module,
                       init: dict) -> None:
    # here, I tell the difference between modules using its name,
    # not type(nn.Module).
    # this may provide some flexibility.
    # for example, when you have two modules of the same type
    # but different configurations inside.
    assert name.lower() == name
    if name.startswith('torch.'):
        _init_mapping_official[name](mod, init)
    else:
        _init_mapping_custom[name](mod, init)
