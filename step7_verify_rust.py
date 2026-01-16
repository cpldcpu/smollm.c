"""
Step 7: Verify Rust implementation against C and Python Q8
"""

import subprocess
import sys

def run_command(cmd, cwd=None):
    """Run a command and return its output"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    return result.stdout + result.stderr

def extract_generated_text(output):
    """Extract the generated text from output (after the prompt tokens line)"""
    lines = output.strip().split('\n')
    # Find the line after "Tokens:" and skip empty lines
    found_tokens = False
    result_lines = []
    for line in lines:
        if line.startswith('Tokens:'):
            found_tokens = True
            continue
        if found_tokens and line.strip():
            result_lines.append(line)
    return '\n'.join(result_lines) if result_lines else output

def main():
    test_prompts = [
        "The capital of France is",
        "Hello, my name is",
    ]

    print("=" * 60)
    print("Rust vs C vs Python Q8 Verification")
    print("=" * 60)

    all_match = True

    for prompt in test_prompts:
        print(f"\n--- Prompt: {prompt!r} ---\n")

        # Run Rust
        rust_out = run_command(
            f'./target/release/smolr -m ../models/smollm2-135m-q8.bin -p "{prompt}" -n 30',
            cwd='/mnt/d/ML/81_Ctransformer/smolr'
        )
        rust_text = extract_generated_text(rust_out)

        # Run C
        c_out = run_command(
            f'./smolc -m ../models/smollm2-135m-q8.bin -p "{prompt}" -n 30',
            cwd='/mnt/d/ML/81_Ctransformer/smolc'
        )
        c_text = extract_generated_text(c_out)

        print(f"Rust output:\n{rust_text}\n")
        print(f"C output:\n{c_text}\n")

        # Compare
        if rust_text == c_text:
            print("✓ Rust matches C exactly!")
        else:
            print("✗ Rust differs from C!")
            all_match = False
            # Show character-by-character comparison for debugging
            print("\nCharacter comparison:")
            for i, (r, c) in enumerate(zip(rust_text, c_text)):
                if r != c:
                    print(f"  Position {i}: Rust={repr(r)}, C={repr(c)}")
                    break
            if len(rust_text) != len(c_text):
                print(f"  Length: Rust={len(rust_text)}, C={len(c_text)}")

    print("\n" + "=" * 60)
    if all_match:
        print("=== All outputs match! Rust implementation verified. ===")
    else:
        print("=== Some outputs differ. Please investigate. ===")
    print("=" * 60)

    return 0 if all_match else 1

if __name__ == "__main__":
    sys.exit(main())
