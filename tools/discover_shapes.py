#!/usr/bin/env python3
"""
Model Shape Discovery Tool for Typo-Fixer CLI

This script discovers model shapes and configurations from CoreML model files,
adapted from the candle-coreml discovery tool for use with typo-fixer models.
"""

import argparse
import json
import os
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import coremltools as ct


def discover_model_config(model_dir: str, verbose: bool = False) -> Dict[str, Any]:
    """Discover model configuration from a directory containing CoreML model files."""
    model_path = Path(model_dir)
    
    if not model_path.exists():
        raise ValueError(f"Model directory does not exist: {model_dir}")
    
    if verbose:
        print(f"🔍 Discovering model shapes in: {model_dir}")
    
    # Find all .mlpackage and .mlmodelc files
    ml_files = []
    for pattern in ["*.mlpackage", "*.mlmodelc"]:
        ml_files.extend(glob.glob(str(model_path / pattern)))
    
    if not ml_files:
        raise ValueError(f"No CoreML model files found in {model_dir}")
    
    if verbose:
        print(f"📁 Found {len(ml_files)} CoreML model files")
        for f in ml_files:
            print(f"   - {Path(f).name}")
    
    # Analyze each model file
    components = {}
    model_info = {
        "path": str(model_path.absolute()),
        "model_type": "qwen",
        "discovered_at": datetime.now().isoformat()
    }
    
    # Inferred shapes (will be updated as we analyze components)
    shapes = {
        "batch_size": 1,
        "context_length": 256,
        "hidden_size": 1024,
        "vocab_size": 151669
    }
    
    for ml_file in ml_files:
        file_path = Path(ml_file)
        component_name = infer_component_type(file_path.name)
        
        if verbose:
            print(f"🔧 Analyzing {file_path.name} as '{component_name}'")
        
        try:
            model = ct.models.MLModel(ml_file)
            spec = model.get_spec()
            
            # Extract input/output information
            inputs = {}
            outputs = {}
            
            # Get inputs
            for input_desc in spec.description.input:
                input_shape = []
                if input_desc.type.HasField('multiArrayType'):
                    for dim in input_desc.type.multiArrayType.shape:
                        input_shape.append(int(dim))
                
                inputs[input_desc.name] = {
                    "name": input_desc.name,
                    "shape": input_shape,
                    "data_type": get_data_type(input_desc.type)
                }
                
                if verbose:
                    print(f"   Input: {input_desc.name} -> {input_shape}")
            
            # Get outputs
            for output_desc in spec.description.output:
                output_shape = []
                if output_desc.type.HasField('multiArrayType'):
                    for dim in output_desc.type.multiArrayType.shape:
                        output_shape.append(int(dim))
                
                outputs[output_desc.name] = {
                    "name": output_desc.name,
                    "shape": output_shape,
                    "data_type": get_data_type(output_desc.type)
                }
                
                if verbose:
                    print(f"   Output: {output_desc.name} -> {output_shape}")
            
            components[component_name] = {
                "file_path": str(file_path.absolute()),
                "inputs": inputs,
                "outputs": outputs,
                "functions": []
            }
            
            # Update shapes based on component analysis
            shapes = update_shapes_from_component(shapes, component_name, inputs, outputs, verbose)
            
        except Exception as e:
            if verbose:
                print(f"⚠️ Warning: Could not analyze {file_path.name}: {e}")
            continue
    
    # Generate naming patterns
    naming = generate_naming_patterns(ml_files)
    
    config = {
        "model_info": model_info,
        "shapes": shapes,
        "components": components,
        "naming": naming
    }
    
    if verbose:
        print(f"✅ Discovery complete")
        print(f"   Detected shapes: {shapes}")
        print(f"   Components: {list(components.keys())}")
    
    return config


def infer_component_type(filename: str) -> str:
    """Infer the component type from the filename."""
    filename_lower = filename.lower()
    
    if "embedding" in filename_lower:
        return "embeddings"
    elif "ffn" in filename_lower and ("prefill" in filename_lower or "pf" in filename_lower):
        return "ffn_prefill"
    elif "ffn" in filename_lower:
        return "ffn_infer"
    elif "lm_head" in filename_lower or "lmhead" in filename_lower:
        return "lm_head"
    else:
        # Try to infer from common patterns
        if filename_lower.startswith("emb"):
            return "embeddings"
        elif "head" in filename_lower:
            return "lm_head"
        else:
            return f"unknown_{filename_lower.replace('.', '_')}"


def get_data_type(type_desc) -> str:
    """Extract data type from CoreML type descriptor."""
    if type_desc.HasField('multiArrayType'):
        data_type = type_desc.multiArrayType.dataType
        if data_type == ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT16:
            return "FLOAT16"
        elif data_type == ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32:
            return "FLOAT32"
        elif data_type == ct.proto.FeatureTypes_pb2.ArrayFeatureType.INT32:
            return "INT32"
        else:
            return "UNKNOWN"
    return "UNKNOWN"


def update_shapes_from_component(shapes: Dict[str, Any], component_name: str, 
                                inputs: Dict, outputs: Dict, verbose: bool = False) -> Dict[str, Any]:
    """Update shape information based on component analysis."""
    new_shapes = shapes.copy()
    
    if component_name == "embeddings":
        # Extract batch_size and context info from embeddings input
        for input_name, input_info in inputs.items():
            if "input_ids" in input_name.lower() and len(input_info["shape"]) >= 2:
                batch_size = input_info["shape"][0]
                context_or_single = input_info["shape"][1]
                
                if batch_size > 0:
                    new_shapes["batch_size"] = batch_size
                
                # If context_or_single is 1, this might be a single-token model
                # If it's larger, it might be the context length
                if context_or_single > 1:
                    new_shapes["context_length"] = context_or_single
                
                if verbose:
                    print(f"   → Updated batch_size: {batch_size}, context hint: {context_or_single}")
        
        # Extract hidden_size from embeddings output
        for output_name, output_info in outputs.items():
            if "hidden" in output_name.lower() and len(output_info["shape"]) >= 3:
                hidden_size = output_info["shape"][-1]  # Last dimension is typically hidden size
                if hidden_size > 0:
                    new_shapes["hidden_size"] = hidden_size
                    if verbose:
                        print(f"   → Updated hidden_size: {hidden_size}")
    
    elif component_name == "lm_head":
        # Extract vocab_size from lm_head outputs
        total_vocab_size = 0
        logits_outputs = 0
        
        for output_name, output_info in outputs.items():
            if "logits" in output_name.lower() and len(output_info["shape"]) >= 3:
                vocab_part_size = output_info["shape"][-1]  # Last dimension
                total_vocab_size += vocab_part_size
                logits_outputs += 1
                
                if verbose:
                    print(f"   → Found logits output '{output_name}': {vocab_part_size} tokens")
        
        if total_vocab_size > 0:
            new_shapes["vocab_size"] = total_vocab_size
            if verbose:
                print(f"   → Updated vocab_size: {total_vocab_size} (from {logits_outputs} parts)")
    
    elif "ffn" in component_name:
        # Extract context_length from causal_mask if available
        for input_name, input_info in inputs.items():
            if "causal_mask" in input_name.lower() and len(input_info["shape"]) >= 4:
                context_length = input_info["shape"][-1]  # Last dimension is context length
                if context_length > new_shapes.get("context_length", 0):
                    new_shapes["context_length"] = context_length
                    if verbose:
                        print(f"   → Updated context_length from causal_mask: {context_length}")
    
    return new_shapes


def generate_naming_patterns(ml_files: List[str]) -> Dict[str, Optional[str]]:
    """Generate file naming patterns from the discovered files."""
    patterns = {
        "embeddings_pattern": None,
        "ffn_infer_pattern": None,
        "ffn_prefill_pattern": None,
        "lm_head_pattern": None
    }
    
    for ml_file in ml_files:
        filename = Path(ml_file).name
        
        if "embedding" in filename.lower():
            patterns["embeddings_pattern"] = filename
        elif "ffn" in filename.lower():
            if "prefill" in filename.lower() or "pf" in filename.lower():
                patterns["ffn_prefill_pattern"] = filename.replace("01of01", "*").replace("_chunk_01", "_chunk_*")
            else:
                patterns["ffn_infer_pattern"] = filename.replace("01of01", "*").replace("_chunk_01", "_chunk_*")
        elif "lm_head" in filename.lower() or "lmhead" in filename.lower():
            patterns["lm_head_pattern"] = filename.replace("lut6", "lut*")
    
    return patterns


def scan_directory_for_models(scan_dir: str, verbose: bool = False) -> List[str]:
    """Scan a directory for subdirectories that contain CoreML model files."""
    scan_path = Path(scan_dir)
    model_dirs = []
    
    if not scan_path.exists():
        raise ValueError(f"Scan directory does not exist: {scan_dir}")
    
    if verbose:
        print(f"🔍 Scanning directory: {scan_dir}")
    
    for item in scan_path.iterdir():
        if item.is_dir():
            # Check if this directory contains .mlpackage or .mlmodelc files
            ml_files = list(item.glob("*.mlpackage")) + list(item.glob("*.mlmodelc"))
            if ml_files:
                model_dirs.append(str(item))
                if verbose:
                    print(f"   📁 Found model directory: {item.name} ({len(ml_files)} files)")
    
    return model_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Discover model shapes and generate configuration files for typo-fixer models"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-dir", help="Path to single model directory to analyze")
    group.add_argument("--scan-directory", help="Path to directory to scan for model subdirectories")
    
    parser.add_argument("--output", help="Output file path (for single model)")
    parser.add_argument("--output-dir", help="Output directory path (for directory scan)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    try:
        if args.model_dir:
            # Single model analysis
            config = discover_model_config(args.model_dir, args.verbose)
            
            if args.output:
                output_path = Path(args.output)
            else:
                model_name = Path(args.model_dir).name
                output_path = Path(f"{model_name}-config.json")
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"✅ Configuration saved to: {output_path}")
            
        else:
            # Directory scan
            model_dirs = scan_directory_for_models(args.scan_directory, args.verbose)
            
            if not model_dirs:
                print(f"⚠️ No model directories found in: {args.scan_directory}")
                return
            
            output_dir = Path(args.output_dir) if args.output_dir else Path("configs")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for model_dir in model_dirs:
                model_name = Path(model_dir).name
                config = discover_model_config(model_dir, args.verbose)
                
                output_path = output_dir / f"{model_name}-config.json"
                with open(output_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                print(f"✅ Configuration saved to: {output_path}")
            
            print(f"🎉 Processed {len(model_dirs)} model directories")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())