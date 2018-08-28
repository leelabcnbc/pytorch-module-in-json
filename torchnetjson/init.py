"""initializers"""

from typing import Callable

from torch import nn

_init_mapping_official = dict()


def _standard_init(mod: nn.Module, init: dict):
    # works for those modules with `weight` and possibly `bias`.
    assert init.keys() == {'strategy', 'parameters'}

    init_inner = {
        'normal': nn.init.normal_,
        'constant': nn.init.constant_,
        'kaiming_normal': nn.init.kaiming_normal_
    }[init['strategy']]

    init_inner(mod.weight, **init['parameters'])

    if mod.bias is not None:
        nn.init.constant_(mod.bias, 0)


def _register_init_official(name: str,
                            init: Callable[[nn.Module, dict], None]) -> None:
    # TODO: more strict check later.
    assert name.lower() == name
    assert name.startswith('torch.')
    _init_mapping_official[name] = init


def _register_init_official_loader() -> None:
    # TODO: make this kind of automatic.
    _register_init_official('torch.nn.conv2d', _standard_init)
    _register_init_official('torch.nn.convtranspose2d', _standard_init)
    _register_init_official('torch.nn.linear', _standard_init)


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
