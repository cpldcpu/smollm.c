"""
Helper script to prepare training data for NoPE continued pretraining.

SmolLM2-135M was pretrained on 2T tokens using:
- FineWeb-Edu (60%): 1.3T educational tokens from web data
- DCLM-Edu (40%): 3.8T tokens filtered for quality Q&A-style content
- The Stack (code data)
- Mathematics and reasoning datasets

For NoPE continued pretraining, we use FineWeb-Edu as the primary dataset
since it matches the original pretraining distribution.

This script can:
1. Download FineWeb-Edu samples from HuggingFace
2. Download sample text data from public sources
3. Clean and prepare text for training
4. Create a combined text file for the training script
"""

import argparse
import requests
from pathlib import Path
import sys

# Check if datasets library is available
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


def download_fineweb_edu(output_file: str, num_samples: int = 1000, max_chars: int = 1000000):
    """
    Download samples from FineWeb-Edu dataset (same as SmolLM2 pretraining)

    FineWeb-Edu is a 1.3T token dataset of educational web content used to train SmolLM2.
    This function downloads a small subset for continued pretraining.
    """
    if not HAS_DATASETS:
        print("Error: 'datasets' library not found. Install with: pip install datasets")
        print("Falling back to synthetic data...")
        return create_synthetic_data(output_file, max_chars)

    print("Downloading FineWeb-Edu samples from HuggingFace...")
    print("This is the same dataset used to pretrain SmolLM2-135M")
    print(f"Fetching {num_samples} samples...")

    try:
        # Load a streaming version to avoid downloading the entire 5.84TB dataset
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",  # 10B token sample
            split="train",
            streaming=True
        )

        all_text = []
        total_chars = 0

        print("Processing samples...")
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break

            text = sample.get('text', '')
            all_text.append(text)
            total_chars += len(text)

            if total_chars >= max_chars:
                print(f"Reached {max_chars} characters, stopping...")
                break

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1} samples, {total_chars} characters")

        combined_text = '\n\n'.join(all_text)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(combined_text)

        print(f"\nSuccessfully downloaded {len(all_text)} samples")
        print(f"Total characters: {total_chars}")
        print(f"Saved to: {output_file}")
        return total_chars

    except Exception as e:
        print(f"Error downloading FineWeb-Edu: {e}")
        print("Falling back to synthetic data...")
        return create_synthetic_data(output_file, max_chars)


def download_wikipedia_sample(output_file: str):
    """Download a sample of Wikipedia text"""
    print("Downloading Wikipedia sample...")

    # Download a simple public domain text (Project Gutenberg example)
    # You can replace this with any text corpus
    urls = [
        "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
        "https://www.gutenberg.org/files/11/11-0.txt",       # Alice in Wonderland
        "https://www.gutenberg.org/files/84/84-0.txt",       # Frankenstein
    ]

    all_text = []
    for url in urls:
        try:
            print(f"Downloading {url}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            text = response.text

            # Basic cleaning - remove Gutenberg header/footer
            lines = text.split('\n')
            # Skip first 50 and last 300 lines (typical Gutenberg headers/footers)
            clean_text = '\n'.join(lines[50:-300])
            all_text.append(clean_text)
            print(f"Downloaded {len(clean_text)} characters")
        except Exception as e:
            print(f"Warning: Failed to download {url}: {e}")

    # Combine all text
    combined_text = '\n\n'.join(all_text)

    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(combined_text)

    print(f"Saved {len(combined_text)} characters to {output_file}")
    return len(combined_text)


def create_synthetic_data(output_file: str, num_chars: int = 100000):
    """Create simple synthetic training data for testing"""
    print(f"Creating synthetic data ({num_chars} characters)...")

    # Simple repeating patterns to test the model
    patterns = [
        "The quick brown fox jumps over the lazy dog. ",
        "Machine learning is a fascinating field of artificial intelligence. ",
        "Natural language processing enables computers to understand human language. ",
        "Deep learning models can learn complex patterns from data. ",
        "Transformers have revolutionized natural language understanding. ",
        "Attention mechanisms allow models to focus on relevant information. ",
        "Positional embeddings help models understand sequence order. ",
        "However, some models work well without explicit positional information. ",
    ]

    text = ""
    while len(text) < num_chars:
        for pattern in patterns:
            text += pattern
            if len(text) >= num_chars:
                break

    text = text[:num_chars]

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"Saved {len(text)} characters to {output_file}")
    return len(text)


def prepare_custom_data(input_files: list, output_file: str):
    """Combine multiple text files into one training file"""
    print(f"Combining {len(input_files)} input files...")

    all_text = []
    total_chars = 0

    for input_file in input_files:
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()
                all_text.append(text)
                total_chars += len(text)
                print(f"Read {len(text)} characters from {input_file}")
        except Exception as e:
            print(f"Warning: Failed to read {input_file}: {e}")

    combined_text = '\n\n'.join(all_text)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(combined_text)

    print(f"Saved {total_chars} total characters to {output_file}")
    return total_chars


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data for NoPE pretraining",
        epilog="""
SmolLM2-135M was pretrained on a mixture of:
  - FineWeb-Edu (60%): Educational web content, 1.3T tokens
  - DCLM-Edu (40%): Quality Q&A-style content, 3.8T tokens
  - The Stack: Code data across 80+ languages
  - Mathematics datasets (InfiMM-WebMath, FineMath)

For best results, use --mode fineweb-edu to match the original pretraining data.
        """
    )

    parser.add_argument("--mode", type=str,
                        choices=["fineweb-edu", "wikipedia", "synthetic", "custom"],
                        default="fineweb-edu",
                        help="Data preparation mode (default: fineweb-edu)")
    parser.add_argument("--output", type=str, default="training_data.txt",
                        help="Output file path")
    parser.add_argument("--input_files", type=str, nargs="+",
                        help="Input files for custom mode")
    parser.add_argument("--num_chars", type=int, default=1000000,
                        help="Maximum number of characters to download (default: 1M)")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples to download from FineWeb-Edu (default: 1000)")

    args = parser.parse_args()

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("NoPE Continued Pretraining - Data Preparation")
    print("=" * 60)

    if args.mode == "fineweb-edu":
        if not HAS_DATASETS:
            print("\nWARNING: 'datasets' library not installed!")
            print("Install with: pip install datasets")
            print("Falling back to synthetic data...\n")
            create_synthetic_data(args.output, args.num_chars)
        else:
            download_fineweb_edu(args.output, args.num_samples, args.num_chars)
    elif args.mode == "wikipedia":
        download_wikipedia_sample(args.output)
    elif args.mode == "synthetic":
        create_synthetic_data(args.output, args.num_chars)
    elif args.mode == "custom":
        if not args.input_files:
            print("Error: --input_files required for custom mode")
            return
        prepare_custom_data(args.input_files, args.output)

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(f"Training data saved to: {args.output}")
    print(f"\nNext steps:")
    print(f"  1. Run training:")
    print(f"     python train_nope.py --data_file {args.output}")
    print(f"  2. Or use the automated pipeline:")
    print(f"     ./run_nope_training.sh {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
