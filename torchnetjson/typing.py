from typing import Union, Tuple, Callable
import torch

io_type = Union[torch.Tensor, Tuple[torch.Tensor, ...]]
op_type = Callable[[io_type], io_type]
op_constructor_type = Callable[..., op_type]
