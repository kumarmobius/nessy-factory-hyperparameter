"""
Unified Scripts Package for NeSy Factory Models (GNN/Language Models + CNN)

This package contains utility scripts and advanced functionality for both GNN/Language Models 
and CNN branches of the NeSy Factory Models library.

Available Script Modules:

GNN/Language Models:
- model_initializer: Advanced model initialization and factory functions
- quick_start_examples: Quick start examples and demonstrations
- list_available_options: System introspection and option listing

CNN Models:
- model_initializer_cnn: Advanced CNN model initialization and factory functions
- quick_start_examples_cnn: Quick start examples and demonstrations for CNNs
- list_available_options_cnn: System introspection and option listing for CNN models

Usage:
    # GNN/Language Model functions
    from scripts.model_initializer import initialize_model_from_yaml
    from scripts.quick_start_examples import main as run_quick_start
    
    # CNN functions  
    from scripts.model_initializer_cnn import initialize_cnn_from_yaml
    from scripts.quick_start_examples_cnn import main as run_quick_start_cnn
"""

import os
import sys
from typing import Any, Dict, List, Optional

# Ensure parent directory is in path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Unified script modules information
SCRIPT_MODULES = {
    # GNN/Language Model modules
    "model_initializer": {
        "description": "Advanced model initialization and factory functions for GNNs and Language Models",
        "main_functions": [
            "initialize_model_from_yaml",
            "initialize_model_from_config_dict", 
            "initialize_model_with_custom_config",
            "initialize_multiple_models",
        ],
        "file": "model_initializer.py",
        "type": "gnn_lm"
    },
    "quick_start_examples": {
        "description": "Quick start examples and common usage patterns for GNNs",
        "main_functions": [
            "quick_example_1_simplest",
            "quick_example_2_custom_config",
            "quick_example_3_with_data",
            "create_model_by_name_and_config",
        ],
        "file": "quick_start_examples.py",
        "type": "gnn_lm"
    },
    "list_available_options": {
        "description": "System introspection and option listing for GNN/LM models",
        "main_functions": [
            "main",
            "print_models_info",
            "print_optimizers_info",
            "print_configurations_info",
        ],
        "file": "list_available_options.py",
        "type": "gnn_lm"
    },
    
    # CNN modules
    "model_initializer_cnn": {
        "description": "Advanced CNN model initialization and factory functions",
        "main_functions": [
            "initialize_cnn_from_yaml",
            "initialize_cnn_from_config_dict",
            "initialize_cnn_with_custom_config", 
            "initialize_multiple_cnns",
        ],
        "file": "model_initializer_cnn.py",
        "type": "cnn"
    },
    "quick_start_examples_cnn": {
        "description": "Quick start examples and common usage patterns for CNNs",
        "main_functions": [
            "cnn_quick_example_1_simplest",
            "cnn_quick_example_2_custom_config",
            "cnn_quick_example_3_with_data",
            "create_cnn_by_name_and_config",
        ],
        "file": "quick_start_examples_cnn.py", 
        "type": "cnn"
    },
    "list_available_options_cnn": {
        "description": "System introspection and option listing for CNN models",
        "main_functions": [
            "main",
            "print_cnn_models_info",
            "print_cnn_optimizers_info",
            "print_cnn_configurations_info",
        ],
        "file": "list_available_options_cnn.py",
        "type": "cnn"
    },
}


def list_available_scripts(script_type: Optional[str] = None):
    """
    List all available script modules and their functions.
    
    Args:
        script_type: Filter by type ('gnn_lm', 'cnn', or None for all)
    """
    if script_type == "gnn_lm":
        print("🔧 Available Script Modules - GNN/Language Models")
        modules = {k: v for k, v in SCRIPT_MODULES.items() if v["type"] == "gnn_lm"}
    elif script_type == "cnn":
        print("🔧 Available Script Modules - CNN Models") 
        modules = {k: v for k, v in SCRIPT_MODULES.items() if v["type"] == "cnn"}
    else:
        print("🔧 Available Script Modules - All (GNN/LM + CNN)")
        modules = SCRIPT_MODULES

    print("=" * 70)

    for script_name, info in modules.items():
        type_badge = "🧠 GNN/LM" if info["type"] == "gnn_lm" else "🖼️ CNN"
        print(f"\n{type_badge} | {script_name.upper().replace('_', ' ')}:")
        print(f"  Description: {info['description']}")
        print(f"  File: {info['file']}")
        print(f"  Key Functions:")
        for func in info["main_functions"]:
            print(f"    - {func}")
        print(f"  Usage: python {info['file']}")


def run_script(script_name: str):
    """
    Run a specific script module.

    Args:
        script_name: Name of the script (any key from SCRIPT_MODULES)
    """
    if script_name not in SCRIPT_MODULES:
        print(f"❌ Unknown script: {script_name}")
        print(f"Available scripts: {list(SCRIPT_MODULES.keys())}")
        return

    script_info = SCRIPT_MODULES[script_name]
    script_type = script_info["type"]
    
    print(f"🚀 Running {script_name.replace('_', ' ').title()} ({script_type.upper()})...")
    print("=" * 50)

    try:
        if script_name == "model_initializer":
            from nesy_factory.scripts import model_initializer
            if hasattr(model_initializer, "demo_all_initialization_methods"):
                model_initializer.demo_all_initialization_methods()
            else:
                print("⚠️  Demo function not found in model_initializer")

        elif script_name == "quick_start_examples":
            from nesy_factory.scripts import quick_start_examples
            if hasattr(quick_start_examples, "main"):
                quick_start_examples.main()
            else:
                print("⚠️  Main function not found in quick_start_examples")

        elif script_name == "list_available_options":
            from nesy_factory.scripts import list_available_options
            if hasattr(list_available_options, "main"):
                list_available_options.main()
            else:
                print("⚠️  Main function not found in list_available_options")

        elif script_name == "model_initializer_cnn":
            from nesy_factory_cnn.scripts import model_initializer_cnn
            if hasattr(model_initializer_cnn, "demo_all_cnn_initialization_methods"):
                model_initializer_cnn.demo_all_cnn_initialization_methods()
            else:
                print("⚠️  Demo function not found in model_initializer_cnn")

        elif script_name == "quick_start_examples_cnn":
            from nesy_factory_cnn.scripts import quick_start_examples_cnn
            if hasattr(quick_start_examples_cnn, "main"):
                quick_start_examples_cnn.main()
            else:
                print("⚠️  Main function not found in quick_start_examples_cnn")

        elif script_name == "list_available_options_cnn":
            from nesy_factory_cnn.scripts import list_available_options_cnn
            if hasattr(list_available_options_cnn, "main"):
                list_available_options_cnn.main()
            else:
                print("⚠️  Main function not found in list_available_options_cnn")

    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        import traceback
        traceback.print_exc()


def run_all_scripts(script_type: Optional[str] = None):
    """
    Run all available script modules.
    
    Args:
        script_type: Filter by type ('gnn_lm', 'cnn', or None for all)
    """
    if script_type == "gnn_lm":
        print("🎯 Running All GNN/Language Model Scripts")
        scripts = [k for k, v in SCRIPT_MODULES.items() if v["type"] == "gnn_lm"]
    elif script_type == "cnn":
        print("🎯 Running All CNN Scripts")
        scripts = [k for k, v in SCRIPT_MODULES.items() if v["type"] == "cnn"]
    else:
        print("🎯 Running All Scripts (GNN/LM + CNN)")
        scripts = list(SCRIPT_MODULES.keys())

    print("=" * 60)

    for script_name in scripts:
        script_type = SCRIPT_MODULES[script_name]["type"]
        type_badge = "🧠" if script_type == "gnn_lm" else "🖼️"
        print(f"\n{'='*20} {type_badge} {script_name.upper().replace('_', ' ')} {'='*20}")
        run_script(script_name)
        print("\n" + "=" * 60)

    print("\n🎉 All scripts completed!")


def get_script_info(script_name: str | None = None, script_type: str | None = None) -> Dict[str, Any]:
    """
    Get information about script modules.

    Args:
        script_name: Specific script name, or None for all scripts
        script_type: Filter by type ('gnn_lm', 'cnn', or None for all)

    Returns:
        Dictionary with script module information
    """
    if script_name is not None:
        if script_name in SCRIPT_MODULES:
            if script_type is None or SCRIPT_MODULES[script_name]["type"] == script_type:
                return {script_name: SCRIPT_MODULES[script_name]}
        return {}
    
    if script_type is not None:
        return {k: v for k, v in SCRIPT_MODULES.items() if v["type"] == script_type}
    
    return SCRIPT_MODULES


def run_all_gnn_scripts():
    """Run all GNN/Language Model scripts."""
    run_all_scripts("gnn_lm")


def run_all_cnn_scripts():
    """Run all CNN scripts."""
    run_all_scripts("cnn")


# Convenience functions for direct script access
def run_model_initializer_demo():
    """Run GNN/Language Model initializer demonstration."""
    run_script("model_initializer")


def run_quick_start():
    """Run GNN/Language Model quick start examples."""
    run_script("quick_start_examples")


def list_options():
    """Run list available options script for GNN/LM models."""
    run_script("list_available_options")


def run_model_initializer_cnn_demo():
    """Run CNN model initializer demonstration."""
    run_script("model_initializer_cnn")


def run_quick_start_cnn():
    """Run CNN quick start examples."""
    run_script("quick_start_examples_cnn")


def list_cnn_options():
    """Run list available options script for CNN models."""
    run_script("list_available_options_cnn")


# Import key functions from both GNN and CNN modules
def import_gnn_functions():
    """Import GNN/Language Model functions into this namespace."""
    try:
        sys.path.insert(0, parent_dir)
        from .model_initializer import (
            initialize_model_from_config_dict,
            initialize_model_from_yaml,
            initialize_model_with_custom_config,
            initialize_multiple_models,
            list_all_available_options,
        )
        from .quick_start_examples import (
            create_model_by_name_and_config,
            quick_example_1_simplest,
            quick_example_2_custom_config,
            quick_example_3_with_data,
        )
        return {
            "initialize_model_from_yaml": initialize_model_from_yaml,
            "initialize_model_from_config_dict": initialize_model_from_config_dict,
            "initialize_model_with_custom_config": initialize_model_with_custom_config,
            "initialize_multiple_models": initialize_multiple_models,
            "list_all_available_options": list_all_available_options,
            "quick_example_1_simplest": quick_example_1_simplest,
            "quick_example_2_custom_config": quick_example_2_custom_config,
            "quick_example_3_with_data": quick_example_3_with_data,
            "create_model_by_name_and_config": create_model_by_name_and_config,
        }
    except ImportError as e:
        print(f"Warning: Could not import GNN functions: {e}")
        return {}


def import_cnn_functions():
    """Import CNN functions into this namespace."""
    try:
        sys.path.insert(0, parent_dir)
        from .model_initializer_cnn import (
            initialize_cnn_from_config_dict,
            initialize_cnn_from_yaml,
            initialize_cnn_with_custom_config,
            initialize_multiple_cnns,
            list_all_available_cnn_options,
        )
        from .quick_start_examples_cnn import (
            create_cnn_by_name_and_config,
            cnn_quick_example_1_simplest,
            cnn_quick_example_2_custom_config,
            cnn_quick_example_3_with_data,
        )
        return {
            "initialize_cnn_from_yaml": initialize_cnn_from_yaml,
            "initialize_cnn_from_config_dict": initialize_cnn_from_config_dict,
            "initialize_cnn_with_custom_config": initialize_cnn_with_custom_config,
            "initialize_multiple_cnns": initialize_multiple_cnns,
            "list_all_available_cnn_options": list_all_available_cnn_options,
            "cnn_quick_example_1_simplest": cnn_quick_example_1_simplest,
            "cnn_quick_example_2_custom_config": cnn_quick_example_2_custom_config,
            "cnn_quick_example_3_with_data": cnn_quick_example_3_with_data,
            "create_cnn_by_name_and_config": create_cnn_by_name_and_config,
        }
    except ImportError as e:
        print(f"Warning: Could not import CNN functions: {e}")
        return {}


# Load functions on import
_gnn_functions = import_gnn_functions()
_cnn_functions = import_cnn_functions()

# Add imported functions to namespace
globals().update(_gnn_functions)
globals().update(_cnn_functions)

# Define what gets exported
__all__ = [
    # Script management
    "list_available_scripts",
    "run_script", 
    "run_all_scripts",
    "get_script_info",
    "run_all_gnn_scripts",
    "run_all_cnn_scripts",
    # Convenience functions - GNN/LM
    "run_model_initializer_demo",
    "run_quick_start", 
    "list_options",
    # Convenience functions - CNN
    "run_model_initializer_cnn_demo",
    "run_quick_start_cnn",
    "list_cnn_options",
    # Constants
    "SCRIPT_MODULES",
]

# Add imported function names to __all__
__all__.extend(_gnn_functions.keys())
__all__.extend(_cnn_functions.keys())

# Package metadata
__version__ = "1.0.0"
__description__ = "Unified utility scripts for NeSy Factory Models (GNN/LM + CNN)"

if __name__ == "__main__":
    # If run directly, show all available scripts
    list_available_scripts()
