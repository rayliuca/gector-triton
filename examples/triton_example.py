#!/usr/bin/env python3
"""
Example script demonstrating GECToRTriton usage.

This script shows how to use GECToR with Triton Inference Server for
grammatical error correction.
"""

import argparse
from transformers import AutoTokenizer
from gector import predict, load_verb_dict

# Import GECToRTriton with fallback
try:
    from gector import GECToRTriton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Warning: tritonclient not installed. Install with: pip install tritonclient[grpc]")


def main():
    parser = argparse.ArgumentParser(
        description='GECToR Triton Inference Example'
    )
    parser.add_argument(
        '--config_path',
        type=str,
        required=True,
        help='Path or model ID to load config from (e.g., gotutiyan/gector-roberta-base-5k)'
    )
    parser.add_argument(
        '--triton_url',
        type=str,
        default='localhost:8001',
        help='Triton server URL (default: localhost:8001)'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='gector',
        help='Model name on Triton server (default: gector)'
    )
    parser.add_argument(
        '--model_version',
        type=str,
        default='1',
        help='Model version on Triton server (default: 1)'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input text file with sentences to correct'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output text file for corrected sentences'
    )
    parser.add_argument(
        '--verb_file',
        type=str,
        default='data/verb-form-vocab.txt',
        help='Path to verb form vocabulary file'
    )
    parser.add_argument(
        '--keep_confidence',
        type=float,
        default=0.0,
        help='Keep confidence bias (default: 0.0)'
    )
    parser.add_argument(
        '--min_error_prob',
        type=float,
        default=0.0,
        help='Minimum error probability threshold (default: 0.0)'
    )
    parser.add_argument(
        '--n_iteration',
        type=int,
        default=5,
        help='Number of iterations (default: 5)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for inference (default: 32)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if not TRITON_AVAILABLE:
        print("ERROR: tritonclient is not installed.")
        print("Install it with: pip install tritonclient[grpc]")
        return 1
    
    # Initialize Triton model
    print(f"Loading model configuration from {args.config_path}...")
    model = GECToRTriton.from_pretrained(
        args.config_path,
        triton_url=args.triton_url,
        model_name=args.model_name,
        model_version=args.model_version,
        verbose=args.verbose
    )
    print(f"Connected to Triton server at {args.triton_url}")
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.config_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.config_path)
    
    # Load verb dictionary
    print(f"Loading verb dictionary from {args.verb_file}...")
    encode, decode = load_verb_dict(args.verb_file)
    
    # Read input
    print(f"Reading input from {args.input}...")
    with open(args.input, 'r') as f:
        srcs = f.read().strip().split('\n')
    print(f"Processing {len(srcs)} sentences...")
    
    # Perform corrections
    corrected = predict(
        model, tokenizer, srcs,
        encode, decode,
        keep_confidence=args.keep_confidence,
        min_error_prob=args.min_error_prob,
        n_iteration=args.n_iteration,
        batch_size=args.batch_size,
    )
    
    # Save output
    print(f"Writing corrected sentences to {args.output}...")
    with open(args.output, 'w') as f:
        f.write('\n'.join(corrected))
    
    print("Done!")
    return 0


if __name__ == '__main__':
    exit(main())
