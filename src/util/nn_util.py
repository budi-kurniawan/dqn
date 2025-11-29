import torch
import torch.nn as nn
from pathlib import Path

def save_params_as_torchscript(model: nn.Module, path: Path) -> None:
    model.eval()  # set to eval mode before exporting
    #example_input = torch.randn(1, 3, 224, 224)

    example_input = torch.tensor([[0.1, 0.2, 0.3, 0.4], [-0.1, 0.2, 0.3, 0.4]])
    traced = torch.jit.trace(model, example_input)
    # if example_input is not None:
    #     # Tracing mode
    #     scripted = torch.jit.trace(model, example_input)
    # else:
    #     # Scripting mode (works with dynamic control flow)
    #     scripted = torch.jit.script(model)
    traced.save(str(path))


def save_params(model: nn.Module, path: Path):
    with path.open("w", encoding="utf-8") as f:
        for name, param in model.named_parameters():
            s = ",".join(map(str, param.data.numpy()))
            f.write(f"{name}:{s}\n")


def save_params_as_onnx(model: nn.Module, path: Path) -> None:
    dummy_input = torch.randn(1, 4)
    onnx_model = torch.onnx.export(
        model,                      # model to export
        dummy_input,                # example input
        dynamo=True
    )
    onnx_model.save(str(path))


def print_params(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        print(f"name:{name}:\nvalues:{param.data}\n")

