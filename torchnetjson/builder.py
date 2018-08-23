import json

from .net import JSONNet


def build_net(param_dict):
    # make sure it's really JSON serializable.
    param_dict_normalized = json.loads(json.dumps(param_dict))
    assert param_dict_normalized == param_dict
    return JSONNet(param_dict_normalized)
