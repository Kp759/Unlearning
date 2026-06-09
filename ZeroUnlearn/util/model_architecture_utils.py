"""
Model architecture extraction utility functions - can be used directly in interactive environments
"""
import json
from typing import Dict, Any
from transformers import AutoModelForCausalLM


def _build_module_tree(modules_list: list) -> Dict[str, Any]:
    """
    Convert flat module list to tree structure
    Each module's submodules are nested in the submodules field
    """
    # Create module dictionary indexed by name
    modules_dict = {}
    for module_info in modules_list:
        name = module_info["name"]
        modules_dict[name] = {
            "type": module_info["type"],
            "parameters": module_info["parameters"],
            "submodules": {}
        }
    
    # Build tree structure: add submodules to parent module's submodules
    # First find the root module
    root_module = None
    if "" in modules_dict:
        root_module = modules_dict[""]
    
    # Sort by hierarchy depth to ensure parent modules are processed first
    # Empty string is special: depth -1 to ensure it's processed first
    def get_depth(name):
        if not name:
            return -1
        return len(name.split('.'))
    
    sorted_items = sorted(modules_dict.items(), key=lambda x: (get_depth(x[0]), x[0]))
    
    for name, module_info in sorted_items:
        if not name:  # Root module, skip (already handled above)
            continue
        
        # Find direct parent module
        parts = name.split('.')
        if len(parts) == 1:
            # Single-level module (e.g., "model"), add directly to root module
            if root_module is not None:
                root_module["submodules"][name] = module_info
        else:
            # Multi-level module (e.g., "model.layers.0")
            parent_name = '.'.join(parts[:-1])
            child_name = parts[-1]
            
            # If parent module exists, add current module to parent's submodules
            if parent_name in modules_dict:
                modules_dict[parent_name]["submodules"][child_name] = module_info
            elif root_module is not None:
                # If parent module doesn't exist, add to root module (fallback)
                root_module["submodules"][name] = module_info
    
    return root_module if root_module else {"modules": list(modules_dict.values())}


def get_model_architecture_dict(model: AutoModelForCausalLM) -> Dict[str, Any]:
    """
    Extract model architecture information as a serializable dictionary (tree structure)
    
    Usage example:
        from util.model_architecture_utils import get_model_architecture_dict
        import json
        
        arch_dict = get_model_architecture_dict(model)
        json.dump(arch_dict, open('model_framework.json', 'w'), indent=4)
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary containing model architecture information (JSON serializable, tree structure)
    """
    architecture = {
        "model_type": model.__class__.__name__,
        "config": {},
        "structure": {},
    }
    
    # Save model configuration information
    try:
        if hasattr(model, 'config'):
            config_dict = model.config.to_dict() if hasattr(model.config, 'to_dict') else dict(model.config)
            architecture["config"] = {
                k: str(v) if not isinstance(v, (int, float, str, bool, list, dict, type(None))) else v
                for k, v in config_dict.items()
            }
    except Exception as e:
        architecture["config"] = {"error": f"Unable to serialize config: {str(e)}"}
    
    # Extract all module information (flat list)
    modules_list = []
    total_params = 0
    trainable_params = 0
    
    for name, module in model.named_modules():
        module_info = {
            "name": name,
            "type": module.__class__.__name__,
            "parameters": {}
        }
        
        # Extract parameter information for this module
        for param_name, param in module.named_parameters(recurse=False):
            module_info["parameters"][param_name] = {
                "shape": list(param.shape),
                "dtype": str(param.dtype),
                "requires_grad": param.requires_grad,
                "numel": param.numel(),
            }
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        # Record module info even if no parameters
        if not module_info["parameters"]:
            module_info["parameters"] = None
        
        modules_list.append(module_info)
    
    # Build tree structure
    architecture["structure"] = _build_module_tree(modules_list)
    
    # Add parameter statistics
    architecture["statistics"] = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "total_parameters_millions": round(total_params / 1e6, 2),
        "trainable_parameters_millions": round(trainable_params / 1e6, 2),
        "num_modules": len(modules_list),
    }
    
    return architecture


def save_model_framework(model: AutoModelForCausalLM, output_path: str, indent: int = 4) -> None:
    """
    Quickly save model architecture to JSON file
    
    Usage example:
        from util.model_architecture_utils import save_model_framework
        save_model_framework(model, 'LLaMA-3.2-1B-framework.json')
    
    Args:
        model: PyTorch model
        output_path: Output file path
        indent: JSON indentation
    """
    arch_dict = get_model_architecture_dict(model)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(arch_dict, f, indent=indent, ensure_ascii=False)
    print(f"Model architecture saved to: {output_path}")
    print(f"Total parameters: {arch_dict['statistics']['total_parameters_millions']:.2f}M")
    print(f"Trainable parameters: {arch_dict['statistics']['trainable_parameters_millions']:.2f}M")
