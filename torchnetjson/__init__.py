from sys import version_info
import torch

assert version_info >= (3, 6)
# TODO: should be relaxed later.
# ideally, we should test against each version in _tested_torch_ver
# _tested_torch_ver = {'0.4.1', '1.0.0', '1.0.1', '1.1.0'}
# assert torch.__version__ in _tested_torch_ver
