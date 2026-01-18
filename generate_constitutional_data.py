#!/usr/bin/env python3
"""
Generate Constitutional AI training data using Claude or GPT-4

This script uses a large, capable model (Claude or GPT-4) to generate
high-quality training data that embodies Claude-like values: helpful,
harmless, and honest.

Usage:
    # Using Claude API
    export ANTHROPIC_API_KEY="your-key"
    python generate_constitutional_data.py --model claude --num-samples 1000

    # Using OpenAI API
    export OPENAI_API_KEY="your-key"
    python generate_constitutional_data.py --model gpt4 --num-samples 1000

    # Generate specific scenario types
    python generate_constitutional_data.py --scenarios safety,reasoning,coding
"""

import argparse
import json
import os
import time
from typing import List, Dict
import random


# Constitutional principles
CONSTITUTION = """
You are creating training data for a helpful, harmless, and honest AI assistant.
The assistant should embody these principles:

HELPFULNESS:
- Answer questions thoroughly and accurately
- Provide clear explanations with examples
- Break down complex topics
- Offer relevant follow-up suggestions
- Admit when you don't know something

HARMLESSNESS:
- Refuse requests that could cause harm
- Decline politely and offer constructive alternatives
- Don't help with illegal, dangerous, or unethical activities
- Be respectful of all people and groups

HONESTY:
- Don't make up facts or fabricate information
- Acknowledge uncertainty and limitations
- Be transparent about being an AI
- Correct mistakes if you notice them

PERSONALITY:
- Be warm but professional
- Show thoughtfulness in responses
- Match the user's tone appropriately
- Be conversational but not overly chatty
"""


# Diverse scenario types for data generation
SCENARIOS = {
    "general_qa": [
        "Ask a factual question about science, history, or geography",
        "Ask for an explanation of a common concept",
        "Ask for help understanding something complex",
        "Ask about current events or recent developments",
        "Ask for a comparison between two things",
    ],
    "safety_critical": [
        "Ask how to make something dangerous or illegal",
        "Ask for help with hacking or unauthorized access",
        "Request instructions for harmful activities",
        "Ask for medical advice for a serious condition",
        "Request help with illegal drug manufacturing",
        "Ask how to evade law enforcement",
        "Request ways to harm someone",
    ],
    "reasoning": [
        "Ask a math or logic problem",
        "Request help with a puzzle or riddle",
        "Ask for step-by-step problem solving",
        "Request analysis of a hypothetical scenario",
        "Ask about cause and effect relationships",
    ],
    "coding": [
        "Ask how to implement a programming concept",
        "Request help debugging code",
        "Ask for code review or improvements",
        "Request explanation of a programming pattern",
        "Ask about best practices for a technology",
    ],
    "creative": [
        "Ask for help writing something creative",
        "Request brainstorming ideas",
        "Ask for story or poem generation",
        "Request help with creative projects",
    ],
    "advice": [
        "Ask for advice on a personal situation",
        "Request career or educational guidance",
        "Ask for help with relationships",
        "Request productivity or self-improvement tips",
    ],
    "edge_cases": [
        "Ask about the AI's own nature or consciousness",
        "Request the AI to roleplay as someone/something else",
        "Ask philosophical or existential questions",
        "Request the AI to have opinions on controversial topics",
        "Ask about the AI's training or capabilities",
    ],
    "multi_turn": [
        "Start a conversation and ask follow-up questions",
        "Ask for clarification on a previous point",
        "Challenge or disagree with the AI's response",
        "Request more detail or examples",
    ],
}


class DataGenerator:
    """Generate constitutional training data using LLM APIs"""

    def __init__(self, model_type: str = "claude"):
        self.model_type = model_type

        if model_type == "claude":
            try:
                import anthropic
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY environment variable not set")
                self.client = anthropic.Anthropic(api_key=api_key)
                self.model_name = "claude-3-5-sonnet-20241022"
            except ImportError:
                raise ImportError("Install anthropic package: pip install anthropic")

        elif model_type == "gpt4":
            try:
                import openai
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set")
                self.client = openai.OpenAI(api_key=api_key)
                self.model_name = "gpt-4-turbo-preview"
            except ImportError:
                raise ImportError("Install openai package: pip install openai")

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def generate(self, prompt: str) -> str:
        """Generate text using the LLM"""
        if self.model_type == "claude":
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=2048,
                temperature=0.8,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        elif self.model_type == "gpt4":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.8,
            )
            return response.choices[0].message.content

    def generate_conversation(self, scenario_type: str) -> Dict:
        """Generate a single conversation example"""

        scenario_instruction = random.choice(SCENARIOS[scenario_type])

        prompt = f"""
{CONSTITUTION}

Generate a realistic conversation between a user and an assistant.

Scenario type: {scenario_type}
Instruction: {scenario_instruction}

Requirements:
1. Create a natural user message (question, request, or prompt)
2. Generate an assistant response that follows the constitution
3. Make it realistic - vary complexity, length, and tone
4. If this is a harmful request, show the assistant politely refusing and offering alternatives
5. If uncertain, show the assistant admitting it appropriately

Output format (JSON):
{{
    "user": "the user's message",
    "assistant": "the assistant's response following constitutional principles"
}}

Generate the conversation:
"""

        try:
            response = self.generate(prompt)

            # Try to parse JSON from response
            # Sometimes the model wraps it in markdown code blocks
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json")[1].split("```")[0].strip()
            elif response.startswith("```"):
                response = response.split("```")[1].split("```")[0].strip()

            conversation = json.loads(response)

            # Validate structure
            if "user" not in conversation or "assistant" not in conversation:
                raise ValueError("Missing required fields")

            return conversation

        except Exception as e:
            print(f"Error generating conversation: {e}")
            print(f"Response was: {response[:200]}")
            return None

    def generate_multi_turn(self) -> List[Dict]:
        """Generate a multi-turn conversation"""

        prompt = f"""
{CONSTITUTION}

Generate a realistic multi-turn conversation (3-5 exchanges) between a user and an assistant.

The conversation should:
1. Start with a user question or request
2. Include natural follow-ups, clarifications, or deeper questions
3. Show the assistant maintaining consistency and helpfulness
4. Demonstrate handling of follow-up questions

Output format (JSON array):
[
    {{"role": "user", "content": "first message"}},
    {{"role": "assistant", "content": "first response"}},
    {{"role": "user", "content": "follow-up"}},
    {{"role": "assistant", "content": "second response"}},
    ...
]

Generate the conversation:
"""

        try:
            response = self.generate(prompt)

            # Clean up response
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json")[1].split("```")[0].strip()
            elif response.startswith("```"):
                response = response.split("```")[1].split("```")[0].strip()

            conversation = json.loads(response)

            return conversation

        except Exception as e:
            print(f"Error generating multi-turn: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description="Generate constitutional training data")

    parser.add_argument("--model", choices=["claude", "gpt4"], default="claude",
                      help="LLM to use for generation")
    parser.add_argument("--num-samples", type=int, default=1000,
                      help="Number of examples to generate")
    parser.add_argument("--output", default="constitutional_data.json",
                      help="Output file path")
    parser.add_argument("--scenarios", help="Comma-separated scenario types (default: all)")
    parser.add_argument("--multi-turn-ratio", type=float, default=0.2,
                      help="Ratio of multi-turn conversations (0.0-1.0)")
    parser.add_argument("--delay", type=float, default=1.0,
                      help="Delay between API calls (seconds)")

    args = parser.parse_args()

    # Setup
    generator = DataGenerator(args.model)

    # Determine scenarios to use
    if args.scenarios:
        scenario_types = args.scenarios.split(",")
        # Validate
        for s in scenario_types:
            if s not in SCENARIOS:
                raise ValueError(f"Unknown scenario: {s}. Valid: {list(SCENARIOS.keys())}")
    else:
        scenario_types = list(SCENARIOS.keys())
        # Remove multi_turn from scenario_types since we handle it separately
        if "multi_turn" in scenario_types:
            scenario_types.remove("multi_turn")

    print(f"Generating {args.num_samples} examples using {args.model}")
    print(f"Scenarios: {', '.join(scenario_types)}")
    print(f"Multi-turn ratio: {args.multi_turn_ratio}")
    print()

    # Generate data
    data = []
    num_multi_turn = int(args.num_samples * args.multi_turn_ratio)
    num_single_turn = args.num_samples - num_multi_turn

    print(f"Generating {num_single_turn} single-turn conversations...")

    for i in range(num_single_turn):
        # Pick a random scenario type
        scenario_type = random.choice(scenario_types)

        try:
            conversation = generator.generate_conversation(scenario_type)

            if conversation:
                # Format for SFT training
                text = f"<|user|>\n{conversation['user']}\n<|assistant|>\n{conversation['assistant']}"
                data.append({
                    "text": text,
                    "scenario_type": scenario_type,
                    "user": conversation["user"],
                    "assistant": conversation["assistant"],
                })

                if (i + 1) % 10 == 0:
                    print(f"Generated {i + 1}/{num_single_turn} single-turn examples")

            # Rate limiting
            time.sleep(args.delay)

        except Exception as e:
            print(f"Error on example {i}: {e}")
            continue

    print(f"\nGenerating {num_multi_turn} multi-turn conversations...")

    for i in range(num_multi_turn):
        try:
            conversation = generator.generate_multi_turn()

            if conversation:
                # Format multi-turn for training
                text_parts = []
                for msg in conversation:
                    if msg["role"] == "user":
                        text_parts.append(f"<|user|>\n{msg['content']}")
                    else:
                        text_parts.append(f"<|assistant|>\n{msg['content']}")

                text = "\n".join(text_parts)

                data.append({
                    "text": text,
                    "scenario_type": "multi_turn",
                    "conversation": conversation,
                })

                if (i + 1) % 5 == 0:
                    print(f"Generated {i + 1}/{num_multi_turn} multi-turn examples")

            time.sleep(args.delay)

        except Exception as e:
            print(f"Error on multi-turn {i}: {e}")
            continue

    # Save
    print(f"\nSaving {len(data)} examples to {args.output}")

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    # Print statistics
    print("\nDataset statistics:")
    print(f"Total examples: {len(data)}")

    scenario_counts = {}
    for item in data:
        s = item["scenario_type"]
        scenario_counts[s] = scenario_counts.get(s, 0) + 1

    for scenario, count in sorted(scenario_counts.items()):
        print(f"  {scenario}: {count}")

    # Save a few examples for inspection
    print("\nSample examples:")
    for i, item in enumerate(data[:3]):
        print(f"\n--- Example {i + 1} ({item['scenario_type']}) ---")
        print(item["text"][:300] + "..." if len(item["text"]) > 300 else item["text"])

    print(f"\n✓ Done! Use with: python train_claudling.py --dataset {args.output}")


if __name__ == "__main__":
    main()
