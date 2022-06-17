import torch
import torch.nn as nn
import torch.utils.tensorboard
import torchinfo
from typing import Union


def test_model(model: nn.Module, input_size: tuple[int, int, int, int] = None,
               input_data: Union[torch.Tensor, list[torch.Tensor]] = None, graph_dir: str = None):
    assert input_size is not None or input_data is not None, 'Either input_size or input_data must specify a value.'

    model.eval()
    model_statistics = torchinfo.summary(
        model,
        input_size,
        input_data,
        depth=10,
        col_names=('input_size', 'kernel_size', 'output_size', 'num_params', 'mult_adds'),
        row_settings=('depth', 'var_names')
    )
    print(f'Total GFLOPs: {model_statistics.total_mult_adds * 2 / 1e9:.4f}')

    if graph_dir is not None:
        writer = torch.utils.tensorboard.SummaryWriter(graph_dir)
        if input_size is not None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            writer.add_graph(model, torch.rand(input_size, device=device))
        else:
            writer.add_graph(model, input_data)
        writer.close()
