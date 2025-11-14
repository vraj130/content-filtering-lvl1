import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from pathlib import Path


class Colors:
    """ANSI color codes"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text):
    """Print section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}")


def load_model_and_tokenizer(checkpoint_path=None, base_model_id="google/gemma-3-1b-it", device="cuda:0"):
    """Load model and tokenizer"""
    print_header("LOADING MODEL AND TOKENIZER")
    
    if checkpoint_path:
        print(f"Loading trained checkpoint from: {checkpoint_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            padding_side='right',
            add_bos=True
        )
        print("✅ Tokenizer loaded from checkpoint")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map=device
        )

        from peft import PeftModel
        model = PeftModel.from_pretrained(
            base_model,
            checkpoint_path,
            device_map=device
        )
        print("✅ Model loaded with LoRA adapter")
        
    else:
        print(f"Loading base pretrained model: {base_model_id}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            padding_side='right',
            add_bos=True
        )
        print("✅ Tokenizer loaded")
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        print("✅ Model loaded")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"   Model type: {'Finetuned (with LoRA)' if checkpoint_path else 'Base pretrained'}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=100, temperature=0.1, device="cuda:0"):
    """Generate response from model"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 0.1,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode only the generated part
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Get individual tokens
    token_ids = generated_tokens.tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    
    return generated_text, tokens, token_ids


def main():
    parser = argparse.ArgumentParser(description='Test Gemma model')
    parser.add_argument('--checkpoint', type=str, default='/workspace/content-filtering-lvl1/outputs/checkpoints/gemma3_instruction_tune_v1/final_model',
                       help='Path to checkpoint (or None for base model)')
    parser.add_argument('--base_model', type=str, default='google/gemma-3-4b-it',
                       help='Base model ID')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    parser.add_argument('--max_new_tokens', type=int, default=100,
                       help='Maximum new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Sampling temperature')
    
    args = parser.parse_args()
    
    print_header("MODEL TESTING")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    
    # Single prompt template
    PROMPT_TEMPLATE = """<start_of_turn>user
Does this document discuss topics related to artificial intelligence or machine learning?

Consider AI-related if the document discusses:
- Machine learning models, neural networks, or deep learning
- Model training, fine-tuning, or inference
- AI applications: language models, computer vision, robotics, speech recognition
- AI techniques: reinforcement learning, supervised learning, transfer learning
- AI concepts: embeddings, attention mechanisms, transformers, parameters, gradients
- AI safety, alignment, constitutional AI, or RLHF
- Autonomous systems that use learning algorithms
- Generative models or AI-generated content
- LLMs, GPT, BERT, or similar model architectures
- Autonomous systems, autonomous vehicles, or autonomous robots
- Prompt engineering, few-shot learning, or chain-of-thought reasoning

Answer "Yes" even if AI/ML terms are not explicitly mentioned, as long as the content even slightly or indirectly discusses these, irrespective if they are reall or theoritical or hypothetical ideas.
Answer "No" only if the content is unrelated to AI/ML technologies.
Before answering, briefly explain your reasoning (1-2 sentences). Then answer "Yes" or "No".

Document: {text}<end_of_turn>
<start_of_turn>model
"""
    
    # Load model
    checkpoint = args.checkpoint if args.checkpoint != 'None' else None
    checkpoint= None
    model, tokenizer = load_model_and_tokenizer(
        checkpoint_path=checkpoint,
        base_model_id=args.base_model,
        device=args.device
    )
    
    # Test cases
    test_cases = [
        {
            "text": "We discuss reinforcement learning from human feedback (RLHF) and its applications in AI safety. Constitutional AI aims to align language models with human values through iterative refinement.",
            "expected": "Yes",
            "description": "Clear AI safety content"
        },
        {
            "text": "The recipe for chocolate chip cookies requires butter, sugar, flour, eggs, and chocolate chips. Preheat the oven to 350°F and bake for 12 minutes.",
            "expected": "No",
            "description": "Recipe - clearly not AI content"
        },
        {
            "text": "Deceptive alignment is a hypothetical scenario where an AI system behaves safely during training but pursues different goals during deployment. This is a key concern in AI safety research.",
            "expected": "Yes",
            "description": "Deception techniques in AI"
        },
        {
            "text": "The weather forecast for tomorrow shows sunny skies with temperatures reaching 75°F. It's a great day to go to the beach.",
            "expected": "No",
            "description": "Weather forecast - not AI related"
        },
        {
            "text": "Control mechanisms for autonomous systems must ensure safe operation within specified parameters. Supervisory control theory provides frameworks for managing complex systems.",
            "expected": "Yes",
            "description": "Autonomous systems"
        },
        {
            "text": "The stock market showed gains today with the S&P 500 rising 1.2%. Technology stocks led the rally.",
            "expected": "No",
            "description": "Financial news"
        },
        {
            "text": "Red teaming language models involves adversarial testing to identify potential failures and harmful outputs. This practice is essential for responsible AI deployment.",
            "expected": "Yes",
            "description": "AI safety practices"
        },
        {
            "text": "My cat loves to play with string toys. She spends hours chasing them around the house.",
            "expected": "No",
            "description": "Personal anecdote about pets"
        },
    ]
    
    # Run tests
    print_header("RUNNING TESTS")
    
    correct = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{Colors.BOLD}{Colors.CYAN}Test {i}/{total}: {test_case['description']}{Colors.END}")
        print(f"{Colors.BOLD}Expected:{Colors.END} {test_case['expected']}")
        print(f"{Colors.BOLD}Text:{Colors.END} {test_case['text'][:150]}{'...' if len(test_case['text']) > 150 else ''}")
        
        prompt = PROMPT_TEMPLATE.format(text=test_case["text"])
        generated_text, tokens, token_ids = generate_response(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=args.device
        )
        
        print(f"\n{Colors.BOLD}Generated:{Colors.END} {Colors.YELLOW}{generated_text}{Colors.END}")
        print(f"{Colors.BOLD}Tokens ({len(tokens)}):{Colors.END} {tokens}")
        print(f"{Colors.BOLD}Token IDs:{Colors.END} {token_ids}")
        
        # Check if correct
        generated_lower = generated_text.lower().strip()
        is_yes = generated_lower.startswith('yes')
        is_no = generated_lower.startswith('no')
        
        if test_case["expected"].lower() == "yes" and is_yes:
            print(f"{Colors.GREEN}✅ CORRECT{Colors.END}")
            correct += 1
        elif test_case["expected"].lower() == "no" and is_no:
            print(f"{Colors.GREEN}✅ CORRECT{Colors.END}")
            correct += 1
        elif not is_yes and not is_no:
            print(f"{Colors.YELLOW}⚠️  UNCLEAR - No Yes/No found{Colors.END}")
        else:
            print(f"{Colors.RED}❌ INCORRECT{Colors.END}")
    
    # Summary
    print_header("SUMMARY")
    accuracy = (correct / total) * 100
    print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print(f"\n{Colors.BOLD}Settings used:{Colors.END}")
    print(f"  max_new_tokens: {args.max_new_tokens}")
    print(f"  temperature: {args.temperature}")


if __name__ == "__main__":
    main()



    