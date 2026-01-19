#!/usr/bin/env python3
"""
GGUF Model Downloader - Search and download GGUF models from Hugging Face

Usage:
    python download_gguf.py                    # Interactive search
    python download_gguf.py "SmolLM2"          # Search with query
    python download_gguf.py --model MODEL_ID   # Direct download
"""

import sys
import os
import argparse
from pathlib import Path

try:
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print("Error: huggingface_hub not installed")
    print("Install with: pip install huggingface-hub")
    sys.exit(1)


def search_gguf_models(query="", limit=20):
    """Search for models with GGUF files on Hugging Face"""
    api = HfApi()
    query_str = f' matching "{query}"' if query else ''
    print(f"Searching Hugging Face for GGUF models{query_str}...")

    # Search for models
    models = api.list_models(
        search=query if query else "gguf",
        cardData=True,
        sort="downloads",
        direction=-1,
        limit=100  # Get more to filter
    )

    gguf_models = []
    for model in models:
        model_id = model.modelId

        # Check if model has GGUF files
        try:
            files = list_repo_files(model_id)
            gguf_files = [f for f in files if f.endswith('.gguf')]

            if gguf_files:
                # Filter for Q8_0 files
                q8_files = [f for f in gguf_files if 'q8_0' in f.lower() or 'q8-0' in f.lower()]

                gguf_models.append({
                    'id': model_id,
                    'downloads': getattr(model, 'downloads', 0),
                    'likes': getattr(model, 'likes', 0),
                    'gguf_files': gguf_files,
                    'q8_files': q8_files
                })

                if len(gguf_models) >= limit:
                    break
        except Exception:
            continue

    return gguf_models


def display_models(models):
    """Display search results"""
    if not models:
        print("\nNo GGUF models found.")
        return

    print(f"\nFound {len(models)} models with GGUF files:\n")
    print(f"{'#':<4} {'Model ID':<50} {'Q8_0':<6} {'Downloads':<12}")
    print("-" * 80)

    for i, model in enumerate(models, 1):
        q8_count = len(model['q8_files'])
        downloads = model['downloads']

        # Format downloads
        if downloads >= 1_000_000:
            dl_str = f"{downloads/1_000_000:.1f}M"
        elif downloads >= 1_000:
            dl_str = f"{downloads/1_000:.1f}K"
        else:
            dl_str = str(downloads)

        q8_indicator = "✓" if q8_count > 0 else "-"
        print(f"{i:<4} {model['id']:<50} {q8_indicator:<6} {dl_str:<12}")


def display_files(model_id, files, q8_only=False):
    """Display GGUF files for a model"""
    if q8_only:
        files = [f for f in files if 'q8_0' in f.lower() or 'q8-0' in f.lower()]

    if not files:
        print(f"\nNo {'Q8_0 ' if q8_only else ''}GGUF files found in {model_id}")
        return []

    print(f"\n{'Q8_0 ' if q8_only else ''}GGUF files in {model_id}:\n")
    print(f"{'#':<4} {'Filename':<60} {'Type':<10}")
    print("-" * 80)

    for i, file in enumerate(files, 1):
        # Extract quant type from filename
        fname = file.lower()
        if 'q8_0' in fname or 'q8-0' in fname:
            qtype = 'Q8_0'
        elif 'q4' in fname:
            qtype = 'Q4'
        elif 'q5' in fname:
            qtype = 'Q5'
        elif 'q6' in fname:
            qtype = 'Q6'
        elif 'f16' in fname or 'fp16' in fname:
            qtype = 'F16'
        elif 'f32' in fname or 'fp32' in fname:
            qtype = 'F32'
        else:
            qtype = 'Unknown'

        print(f"{i:<4} {file:<60} {qtype:<10}")

    return files


def download_file(model_id, filename, output_dir="models"):
    """Download a specific file from a model"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {filename} from {model_id}...")
    print(f"Destination: {output_path / filename}")

    try:
        local_path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            local_dir=output_path,
            local_dir_use_symlinks=False
        )
        print(f"✓ Downloaded successfully: {local_path}")
        return local_path
    except HfHubHTTPError as e:
        print(f"✗ Download failed: {e}")
        return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def interactive_search():
    """Interactive search and download workflow"""
    query = input("Enter search query (or press Enter for popular GGUF models): ").strip()

    models = search_gguf_models(query, limit=15)

    if not models:
        return

    display_models(models)

    while True:
        try:
            choice = input("\nSelect a model number (or 'q' to quit): ").strip()

            if choice.lower() == 'q':
                return

            idx = int(choice) - 1
            if 0 <= idx < len(models):
                model = models[idx]
                break
            else:
                print(f"Invalid selection. Please enter 1-{len(models)}")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Show files
    model_id = model['id']

    # Prefer Q8_0 files
    if model['q8_files']:
        print(f"\n✓ Found Q8_0 files (recommended for smolc)")
        files = display_files(model_id, model['q8_files'], q8_only=False)
    else:
        print(f"\n⚠ No Q8_0 files found. Showing all GGUF files:")
        print("  Note: gguf_to_smol only supports Q8_0 quantization")
        files = display_files(model_id, model['gguf_files'], q8_only=False)

    if not files:
        return

    # Select file
    while True:
        try:
            file_choice = input("\nSelect a file number to download (or 'q' to quit): ").strip()

            if file_choice.lower() == 'q':
                return

            file_idx = int(file_choice) - 1
            if 0 <= file_idx < len(files):
                filename = files[file_idx]
                break
            else:
                print(f"Invalid selection. Please enter 1-{len(files)}")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Download
    local_path = download_file(model_id, filename)

    if local_path:
        print("\n" + "="*80)
        print("Next steps:")
        print(f"1. Convert to SMOL format:")
        print(f"   ./smolc/gguf_to_smol {local_path} models/model.bin")
        print(f"2. Run inference:")
        print(f"   ./smolc/smolc -m models/model.bin -p \"Your prompt\" -n 50")
        print("="*80)


def direct_download(model_id, filename=None, output_dir="models"):
    """Direct download of a model"""
    try:
        files = list_repo_files(model_id)
        gguf_files = [f for f in files if f.endswith('.gguf')]

        if not gguf_files:
            print(f"No GGUF files found in {model_id}")
            return

        # If no filename specified, try to find Q8_0
        if not filename:
            q8_files = [f for f in gguf_files if 'q8_0' in f.lower() or 'q8-0' in f.lower()]

            if q8_files:
                if len(q8_files) == 1:
                    filename = q8_files[0]
                    print(f"Found Q8_0 file: {filename}")
                else:
                    print("Multiple Q8_0 files found:")
                    display_files(model_id, q8_files)
                    return
            else:
                print("No Q8_0 files found. Available GGUF files:")
                display_files(model_id, gguf_files)
                return

        download_file(model_id, filename, output_dir)

    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Search and download GGUF models from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_gguf.py                          # Interactive search
  python download_gguf.py "SmolLM2"                # Search for SmolLM2
  python download_gguf.py --model bartowski/SmolLM2-135M-Instruct-GGUF
  python download_gguf.py --model bartowski/SmolLM2-135M-Instruct-GGUF --file smollm2-135m-instruct-q8_0.gguf
        """
    )

    parser.add_argument('query', nargs='?', help='Search query')
    parser.add_argument('--model', '-m', help='Direct model ID to download')
    parser.add_argument('--file', '-f', help='Specific file to download')
    parser.add_argument('--output', '-o', default='models', help='Output directory (default: models)')

    args = parser.parse_args()

    if args.model:
        # Direct download mode
        direct_download(args.model, args.file, args.output)
    elif args.query:
        # Search with query
        models = search_gguf_models(args.query, limit=15)
        display_models(models)

        if models:
            print("\nRun again and select a model number to download,")
            print("or use: python download_gguf.py --model MODEL_ID")
    else:
        # Interactive mode
        interactive_search()


if __name__ == "__main__":
    main()
