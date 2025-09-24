import torch
import torch.nn as nn
from pathlib import Path

def save_params_as_torchscript(model: nn.Module, path: Path, example_input: torch.Tensor = None) -> None:
    """
    Save a PyTorch nn.Module as a TorchScript file.

    Args:
        model (nn.Module): The model to export.
        path (Path): Destination file path (e.g., Path("model.pt")).
        example_input (torch.Tensor, optional): 
            Example input for tracing. If None, use scripting instead.
    """
    model.eval()  # set to eval mode before exporting
    
    if example_input is not None:
        # Tracing mode
        scripted = torch.jit.trace(model, example_input)
    else:
        # Scripting mode (works with dynamic control flow)
        scripted = torch.jit.script(model)
    scripted.save(str(path))


def save_params(model: nn.Module, path: Path):
    with path.open("w", encoding="utf-8") as f:
        for name, param in model.named_parameters():
            s = ",".join(map(str, param.data.numpy()))
            f.write(f"{name}:{s}\n")


