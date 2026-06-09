"""
Utility script: Extract and save model architecture information to JSON format
"""
import json
import torch
from typing import Dict, Any, List
from transformers import AutoModelForCausalLM


def save_model_architecture(
    model: AutoModelForCausalLM,
    output_path: str,
    indent: int = 4
) -> None:
    """
    Extract model architecture and save to JSON file (tree structure)
    
    Args:
        model: PyTorch model
        output_path: Output JSON file path
        indent: JSON indentation spaces
    """
    # Import utility functions
    from .model_architecture_utils import get_model_architecture_dict
    
    print(f"Extracting model architecture information...")
    architecture = get_model_architecture_dict(model)
    
    print(f"Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(architecture, f, indent=indent, ensure_ascii=False)
    
    print(f"Model architecture saved to {output_path}")
    print(f"Total parameters: {architecture['statistics']['total_parameters_millions']:.2f}M")
    print(f"Trainable parameters: {architecture['statistics']['trainable_parameters_millions']:.2f}M")
    print(f"Number of modules: {architecture['statistics']['num_modules']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract and save model architecture information")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name or path (e.g., 'meta-llama/Llama-3.2-1B')"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: model_name_framework.json)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        device_map="auto" if args.device == "cuda" else None
    )
    if args.device == "cpu":
        model = model.to("cpu")
    
    # Determine output path
    if args.output is None:
        model_name_safe = args.model_name.replace("/", "_").replace("-", "_")
        args.output = f"{model_name_safe}_framework.json"
    
    # Save architecture information
    save_model_architecture(model, args.output)
