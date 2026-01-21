"""
Helper script to prepare training data for NoPE continued pretraining.

This script can:
1. Download sample text data from various sources
2. Clean and prepare text for training
3. Create a combined text file for the training script
"""

import argparse
import requests
from pathlib import Path


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
    parser = argparse.ArgumentParser(description="Prepare training data for NoPE pretraining")

    parser.add_argument("--mode", type=str, choices=["wikipedia", "synthetic", "custom"],
                        default="synthetic",
                        help="Data preparation mode")
    parser.add_argument("--output", type=str, default="training_data.txt",
                        help="Output file path")
    parser.add_argument("--input_files", type=str, nargs="+",
                        help="Input files for custom mode")
    parser.add_argument("--num_chars", type=int, default=100000,
                        help="Number of characters for synthetic data")

    args = parser.parse_args()

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "wikipedia":
        download_wikipedia_sample(args.output)
    elif args.mode == "synthetic":
        create_synthetic_data(args.output, args.num_chars)
    elif args.mode == "custom":
        if not args.input_files:
            print("Error: --input_files required for custom mode")
            return
        prepare_custom_data(args.input_files, args.output)

    print(f"\nData preparation complete! Training data saved to: {args.output}")
    print(f"\nYou can now run training with:")
    print(f"  python train_nope.py --data_file {args.output}")


if __name__ == "__main__":
    main()
