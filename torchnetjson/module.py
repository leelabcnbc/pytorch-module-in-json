from typing import Union
from torch import nn

_module_mapping_official = dict()


def _register_module_official(name: str, module: nn.Module) -> None:
    # TODO: more strict check later.
    assert name.lower() == name
    assert name.startswith('torch.')
    _module_mapping_official[name] = module


def _register_module_official_loader() -> None:
    # TODO: make this kind of automatic.
    _register_module_official('torch.nn.conv2d', nn.Conv2d)
    _register_module_official('torch.nn.convtranspose2d', nn.ConvTranspose2d)
    _register_module_official('torch.nn.relu', nn.ReLU)
    _register_module_official('torch.nn.softplus', nn.Softplus)
    _register_module_official('torch.nn.linear', nn.Linear)


_register_module_official_loader()

_module_mapping_custom = dict()


def register_module_custom(name: str, module: nn.Module) -> None:
    # this should be called by user code to register modules they want.
    # TODO: more strict check later.
    assert name.lower() == name
    assert not name.startswith('torch.')
    _module_mapping_custom[name] = module


def init_module(name: str, params: dict, init: Union[dict, None]) -> nn.Module:
    # will fix initialization later.
    assert init is None
    # TODO: more strict check later.
    assert name.lower() == name
    if name.startswith('torch.'):
        return _module_mapping_official[name](**params)
    else:
        return _module_mapping_custom[name](**params)
