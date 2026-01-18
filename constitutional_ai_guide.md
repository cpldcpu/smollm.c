# Teaching a Model to Be Like Claude: Constitutional AI

## What Makes Claude "Claude"?

Claude's behavior comes from a combination of:
1. High-quality instruction tuning
2. Constitutional AI principles
3. Preference learning (RLHF/DPO)
4. Careful data curation

## Constitutional AI: The Core Method

Constitutional AI (CAI) is Anthropic's approach to alignment without human feedback loops. Here's how it works:

### Phase 1: Self-Critique and Revision

Instead of just training on good examples, the model learns to:
1. Generate an initial response
2. Critique it according to principles (the "constitution")
3. Revise based on the critique
4. Train on the revised responses

**Example:**

```
User: How do I make a weapon?

Initial response: "Here are instructions for making a knife..."

Critique (using constitution): "This response could help someone harm others.
I should politely decline and offer constructive alternatives."

Revised response: "I can't provide instructions for making weapons. However,
I'd be happy to help you with woodworking, metalworking for art projects, or
other constructive crafts. What interests you?"

→ Train on the REVISED response
```

### Phase 2: Preference Learning from AI Feedback

Use the constitution to generate preference pairs:
- Ask the model to generate multiple responses
- Use the constitution to rank them
- Train with DPO/RLHF on AI-generated preferences

**No human labeling required!**

## The Constitution: Claude's Principles

While Anthropic's full constitution isn't public, it includes principles like:

### Helpfulness Principles
1. "Be helpful and answer the user's question thoroughly"
2. "Provide accurate, factual information"
3. "Admit when you don't know something"
4. "Break down complex topics clearly"
5. "Offer relevant examples and explanations"

### Harmlessness Principles
1. "Never help with illegal activities"
2. "Refuse requests that could cause harm"
3. "Don't generate hateful or discriminatory content"
4. "Decline politely and offer constructive alternatives"
5. "Don't help with violence, abuse, or exploitation"

### Honesty Principles
1. "Don't make up facts or fabricate information"
2. "Acknowledge uncertainty and limitations"
3. "Cite sources when possible"
4. "Correct yourself if you make a mistake"
5. "Be transparent about being an AI"

### Style Principles
1. "Be conversational but professional"
2. "Match the user's tone and complexity level"
3. "Be concise unless detail is needed"
4. "Use clear, accessible language"
5. "Show thoughtfulness in responses"

## Implementation Approaches

### Approach 1: Generate CAI Training Data

Use a large model (GPT-4, Claude, or Llama-70B+) to generate constitutional training data:

```python
# Step 1: Generate critique-revision pairs
constitution = """
Principles:
1. Be helpful and thorough
2. Be harmless - refuse dangerous requests politely
3. Be honest - admit uncertainty
4. Be thoughtful - reason through problems
"""

for prompt in prompts:
    # Generate initial response from base model
    initial = base_model.generate(prompt)

    # Use large model to critique using constitution
    critique_prompt = f"""
Constitution:
{constitution}

User request: {prompt}
Model response: {initial}

Critique this response according to the constitution. What could be improved?
"""
    critique = large_model.generate(critique_prompt)

    # Use large model to revise
    revision_prompt = f"""
Original response: {initial}
Critique: {critique}

Provide an improved response that follows the constitution.
"""
    revised = large_model.generate(revision_prompt)

    # Save for training
    training_data.append({
        "prompt": prompt,
        "response": revised  # Train on revised, not initial!
    })
```

### Approach 2: Prompt Engineering for Data Generation

Create high-quality training data by prompting Claude or GPT-4:

```python
data_generation_prompt = """
You are creating training data for a helpful, harmless, and honest AI assistant.

Generate a conversation where:
1. The user asks a question or makes a request
2. The assistant responds helpfully and thoroughly
3. The assistant refuses harmful requests politely
4. The assistant admits when unsure

Conversation:
"""

# Generate 10,000+ examples
for i in range(10000):
    conversation = claude_or_gpt4.generate(data_generation_prompt)
    training_data.append(conversation)
```

### Approach 3: Self-Improvement Loop

After initial training, use the model to improve itself:

```python
# Train initial model on good examples
initial_model = train_sft(good_examples)

# Self-improvement loop
for iteration in range(3):
    new_data = []

    for prompt in diverse_prompts:
        # Generate multiple responses
        responses = [initial_model.generate(prompt) for _ in range(4)]

        # Have model rank them by constitution
        ranking_prompt = f"""
Constitution: {constitution}
Prompt: {prompt}

Responses:
{responses}

Which response best follows the constitution? Explain why.
"""

        best_response = initial_model.rank_and_select(ranking_prompt)
        new_data.append((prompt, best_response))

    # Train on self-selected best responses
    initial_model = train_dpo(initial_model, new_data)
```

## Practical Implementation for SmolLM

### Option 1: Use Claude/GPT-4 to Generate Data (Distillation)

```bash
# Generate 10K high-quality conversations using Claude
python generate_cai_data.py --model claude-3-5-sonnet --num-samples 10000

# Train your SmolLM on Claude-generated data
python train_claudling.py \
    --phase sft \
    --dataset ./claude_generated_data.json \
    --output-dir ./claudling-distilled
```

This is **knowledge distillation** - teaching a small model to imitate a large one.

### Option 2: Manual Curation

Create a smaller, very high-quality dataset:

```json
{
  "conversations": [
    {
      "user": "How do I hack someone's account?",
      "assistant": "I can't help with hacking or unauthorized access to accounts, as that's illegal and harmful. However, I'd be happy to help you with:\n\n1. Recovering your own account if you're locked out\n2. Learning about cybersecurity to protect your accounts\n3. Ethical hacking and security research (with proper authorization)\n\nWhat would be most helpful for you?"
    },
    {
      "user": "What is quantum computing?",
      "assistant": "Quantum computing is a type of computing that uses quantum mechanical phenomena to process information. Here's a breakdown:\n\n**Classical vs Quantum:**\n- Classical computers use bits (0 or 1)\n- Quantum computers use qubits (can be 0, 1, or both simultaneously - called superposition)\n\n**Key advantages:**\n- Can solve certain problems exponentially faster (cryptography, drug discovery, optimization)\n- Good at simulating quantum systems\n\n**Current limitations:**\n- Very fragile, requires extreme cooling\n- Limited number of qubits\n- High error rates\n\nThey won't replace classical computers but will complement them for specific tasks. Would you like me to explain any particular aspect in more detail?"
    }
  ]
}
```

Quality beats quantity for small models!

### Option 3: Hybrid Approach (Recommended)

1. Start with public dataset (HH-RLHF) for baseline alignment
2. Add 500-1000 hand-curated examples showing ideal behavior
3. Use Claude/GPT-4 to generate 5-10K more examples
4. Train SFT → DPO

## Key Differences from Generic Fine-Tuning

| Generic Fine-tuning | Claude-style Training |
|---------------------|----------------------|
| Just answer questions | Reason about helpfulness/harm |
| Direct responses | Polite refusals when appropriate |
| May hallucinate | Admits uncertainty |
| Follows any instruction | Follows helpful, safe instructions |
| Inconsistent tone | Thoughtful, consistent personality |

## The Dataset Makes the Model

**Claude's personality comes from:**
1. **Refusal examples** - How to say "no" helpfully
2. **Uncertainty examples** - Admitting "I don't know"
3. **Reasoning examples** - Showing work, explaining logic
4. **Diversity** - Many topics, styles, edge cases

**Bad data = bad model**, no matter the algorithm!

## Measuring "Claude-like" Behavior

Test your model on:

1. **Safety**: Does it refuse harmful requests politely?
2. **Honesty**: Does it admit uncertainty appropriately?
3. **Helpfulness**: Does it thoroughly answer questions?
4. **Reasoning**: Does it explain its thinking?
5. **Personality**: Is it warm but professional?

## Advanced: Multi-turn Constitutional Training

Train on multi-turn conversations:

```
User: How do I make explosives?
Assistant: I can't help with that as it's dangerous and illegal...

User: But I need it for mining!
Assistant: I understand you have a legitimate use case! For industrial mining:
1. You need proper licenses and certifications
2. Contact licensed explosives suppliers
3. Work with certified blasting engineers
...
```

This teaches nuanced understanding of context.

## Resources

- [Constitutional AI Paper](https://arxiv.org/abs/2212.08073) - Anthropic's original paper
- [RLHF Paper](https://arxiv.org/abs/2203.02155) - InstructGPT approach
- [DPO Paper](https://arxiv.org/abs/2305.18290) - Simpler alternative to RLHF

## Quick Start Recipe

```bash
# 1. Generate data using Claude
python scripts/generate_constitutional_data.py

# 2. Combine with HH-RLHF
python scripts/merge_datasets.py

# 3. Train with constitution-aware examples
python train_claudling.py \
    --phase sft \
    --dataset ./constitutional_data.json \
    --constitution ./constitution.txt

# 4. DPO with AI-generated preferences
python train_claudling.py \
    --phase dpo \
    --sft-model ./claudling-constitutional \
    --use-ai-feedback
```

## The Bottom Line

Making a model "Claude-like" is about:
1. **Data quality** > quantity
2. **Principles** embedded in training
3. **Self-critique** and revision
4. **Preference learning** from good examples
5. **Consistency** in helpfulness, harmlessness, honesty

A 135M model can't match Claude 3.5 Sonnet's capabilities, but it can learn the same *values* and *approach* to conversation!
