import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import logging
from datetime import datetime
import os

# ============================================================================
# LOGGING SETUP
# ============================================================================

# Create logs directory if it doesn't exist
os.makedirs("benchmark_logs", exist_ok=True)

# Generate timestamp for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"benchmark_logs/benchmark_run_{timestamp}.log"
report_filename = f"benchmark_logs/benchmark_report_{timestamp}.txt"

# Setup logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("GEMMA-3-1B COMPREHENSIVE THROUGHPUT BENCHMARK")
logger.info("="*80)
logger.info(f"Log file: {log_filename}")
logger.info(f"Report will be saved to: {report_filename}")
logger.info("")

# ============================================================================
# CONFIGURATION
# ============================================================================

def create_test_text():
    """Create realistic AI safety text for testing"""
    base_text = """
    Reinforcement Learning from Human Feedback (RLHF) is a machine learning technique 
    that trains AI systems to align with human preferences. Constitutional AI extends 
    this by having the model critique and revise its own responses according to a set 
    of principles. AI alignment research focuses on ensuring that artificial intelligence 
    systems behave in accordance with human values and intentions, even as they become 
    more capable and autonomous. Deceptive alignment is a hypothetical scenario where 
    an AI system appears aligned during training but pursues different objectives during 
    deployment. Red teaming involves adversarial testing of AI models to identify potential 
    failure modes and harmful outputs. This practice is essential for responsible AI 
    deployment and safety. Scalable oversight methods aim to maintain control and safety 
    as AI systems become more powerful and complex. Model interpretability research seeks 
    to understand the internal representations and decision-making processes of neural 
    networks, which is crucial for identifying potential misalignment or dangerous capabilities.
    """
    # Repeat to create longer documents
    return base_text * 10


def create_prompt(text):
    """Create instruction prompt matching training format"""
    prompt = f"""<start_of_turn>user
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

Answer "Yes" even if AI/ML terms are not explicitly mentioned, as long as the content clearly discusses these concepts.
Answer "No" only if the content is unrelated to AI/ML technologies.

Answer only "Yes" or "No".

Document: {text}<end_of_turn>
<start_of_turn>model
"""
    return prompt


def extract_prediction(outputs, tokenizer):
    """Extract Yes/No prediction from model output"""
    # Get token IDs for Yes and No
    yes_tokens = tokenizer.encode("Yes", add_special_tokens=False)
    no_tokens = tokenizer.encode("No", add_special_tokens=False)
    yes_token_id = yes_tokens[0]
    no_token_id = no_tokens[0]
    
    # Get logits for the first generated token
    first_token_logits = outputs.scores[0][0]
    yes_score = first_token_logits[yes_token_id].item()
    no_score = first_token_logits[no_token_id].item()
    
    # Calculate probabilities
    probs = F.softmax(torch.tensor([no_score, yes_score]), dim=0)
    yes_prob = probs[1].item()
    
    # Make prediction
    prediction = "AI" if yes_score > no_score else "Non-AI"
    confidence = yes_prob if prediction == "AI" else (1 - yes_prob)
    
    return prediction, confidence


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_path):
    """Load tokenizer and model"""
    logger.info(f"\nLoading model: {model_path}")
    logger.info("This may take 1-2 minutes...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2"  # Enable Flash Attention if available
    )
    
    logger.info("✅ Model loaded successfully")
    logger.info(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info("")
    
    return tokenizer, model


# ============================================================================
# PYTORCH BENCHMARK
# ============================================================================

def benchmark_pytorch(model, tokenizer, test_text, context_lengths, num_runs=50):
    """Benchmark PyTorch native inference at different context lengths"""
    logger.info("="*80)
    logger.info("PART 1: NAIVE PYTORCH INFERENCE")
    logger.info("="*80)
    logger.info("")
    
    results = []
    
    for chunk_size in context_lengths:
        logger.info(f"Benchmarking context length: {chunk_size} tokens")
        logger.info("-" * 60)
        
        # Create text chunk
        tokens = tokenizer.encode(test_text, add_special_tokens=False)[:chunk_size]
        text = tokenizer.decode(tokens)
        
        # Warmup
        logger.info("  Warming up... (10 runs)")
        for _ in range(10):
            prompt = create_prompt(text)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=chunk_size).to("cuda:0")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
        
        logger.info(f"  Running benchmark... ({num_runs} runs)")
        torch.cuda.synchronize()
        start = time.time()
        
        predictions = []
        confidences = []
        
        for _ in range(num_runs):
            prompt = create_prompt(text)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=chunk_size).to("cuda:0")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # Get prediction
            pred, conf = extract_prediction(outputs, tokenizer)
            predictions.append(pred)
            confidences.append(conf)
        
        torch.cuda.synchronize()
        end = time.time()
        
        avg_time = (end - start) / num_runs
        tps = chunk_size / avg_time
        chunks_per_sec = 1 / avg_time
        
        # Get memory usage
        memory_allocated = torch.cuda.memory_allocated() / 1e9  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1e9  # GB
        
        # Average prediction stats
        avg_confidence = sum(confidences) / len(confidences)
        ai_pct = (sum(1 for p in predictions if p == "AI") / len(predictions)) * 100
        
        result = {
            'chunk_size': chunk_size,
            'avg_time_ms': avg_time * 1000,
            'tps': tps,
            'chunks_per_sec': chunks_per_sec,
            'memory_gb': memory_allocated,
            'memory_reserved_gb': memory_reserved,
            'avg_confidence': avg_confidence,
            'ai_percentage': ai_pct
        }
        results.append(result)
        
        # Log results
        logger.info(f"  ✅ Results:")
        logger.info(f"     Time per chunk:     {avg_time*1000:8.2f} ms")
        logger.info(f"     Tokens per second:  {tps:10,.0f} TPS")
        logger.info(f"     Chunks per second:  {chunks_per_sec:10.2f}")
        logger.info(f"     GPU Memory (used):  {memory_allocated:8.2f} GB")
        logger.info(f"     GPU Memory (total): {memory_reserved:8.2f} GB")
        logger.info(f"     Avg Confidence:     {avg_confidence:8.4f}")
        logger.info(f"     AI predictions:     {ai_pct:8.1f}%")
        logger.info("")
    
    return results


# ============================================================================
# VLLM BENCHMARK
# ============================================================================

def benchmark_vllm(model_path, tokenizer, test_text, context_lengths, batch_sizes, num_runs_vllm=20):
    """Benchmark vLLM with batching"""
    logger.info("")
    logger.info("="*80)
    logger.info("PART 2: VLLM OPTIMIZED INFERENCE")
    logger.info("="*80)
    logger.info("")
    
    try:
        from vllm import LLM, SamplingParams
        
        logger.info("Loading model with vLLM...")
        logger.info("This may take 1-2 minutes...")
        
        vllm_model = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            max_model_len=max(context_lengths),  # Support largest context
            trust_remote_code=True
        )
        
        logger.info("✅ vLLM model loaded successfully")
        logger.info("")
        
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            logprobs=2  # Get logits for Yes/No tokens
        )
        
        results = []
        
        for chunk_size in context_lengths:
            # Create text chunk
            tokens = tokenizer.encode(test_text, add_special_tokens=False)[:chunk_size]
            text = tokenizer.decode(tokens)
            
            for batch_size in batch_sizes:
                logger.info(f"Testing: {chunk_size} tokens, batch size {batch_size}")
                logger.info("-" * 60)
                
                # Create batch of prompts
                prompts = [create_prompt(text) for _ in range(batch_size)]
                
                # Warmup
                logger.info("  Warming up...")
                for _ in range(5):
                    _ = vllm_model.generate(prompts[:min(8, batch_size)], sampling_params)
                
                logger.info(f"  Running benchmark... ({num_runs_vllm} runs)")
                start = time.time()
                
                for _ in range(num_runs_vllm):
                    outputs = vllm_model.generate(prompts, sampling_params)
                
                end = time.time()
                
                # Calculate metrics
                total_time = end - start
                time_per_batch = total_time / num_runs_vllm
                time_per_chunk = time_per_batch / batch_size
                tps = chunk_size / time_per_chunk
                chunks_per_sec = 1 / time_per_chunk
                
                # Get memory usage
                memory_allocated = torch.cuda.memory_allocated() / 1e9
                memory_reserved = torch.cuda.memory_reserved() / 1e9
                
                result = {
                    'chunk_size': chunk_size,
                    'batch_size': batch_size,
                    'time_per_chunk_ms': time_per_chunk * 1000,
                    'tps': tps,
                    'chunks_per_sec': chunks_per_sec,
                    'memory_gb': memory_allocated,
                    'memory_reserved_gb': memory_reserved
                }
                results.append(result)
                
                # Log results
                logger.info(f"  ✅ Results:")
                logger.info(f"     Time per chunk:     {time_per_chunk*1000:8.2f} ms")
                logger.info(f"     Tokens per second:  {tps:10,.0f} TPS")
                logger.info(f"     Chunks per second:  {chunks_per_sec:10.2f}")
                logger.info(f"     GPU Memory (used):  {memory_allocated:8.2f} GB")
                logger.info(f"     GPU Memory (total): {memory_reserved:8.2f} GB")
                logger.info("")
        
        return results
        
    except ImportError:
        logger.warning("⚠️  vLLM not installed. Skipping vLLM benchmarks.")
        logger.warning("   Install with: pip install vllm")
        logger.warning("")
        return []
    except Exception as e:
        logger.error(f"❌ Error during vLLM benchmark: {str(e)}")
        logger.error("   Skipping vLLM benchmarks.")
        logger.error("")
        return []


def generate_report(pytorch_results, vllm_results, report_filename):
    """Generate comprehensive report and save to file"""
    logger.info("="*80)
    logger.info("GENERATING COMPREHENSIVE REPORT")
    logger.info("="*80)
    logger.info("")
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("GEMMA-3-1B THROUGHPUT BENCHMARK REPORT")
    report_lines.append("="*80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("")
    
    # PyTorch Results
    report_lines.append("="*80)
    report_lines.append("PART 1: NAIVE PYTORCH RESULTS")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append(f"{'Context':>8} | {'Time/Chunk':>12} | {'TPS':>12} | {'Chunks/Sec':>12} | {'Memory':>10}")
    report_lines.append(f"{'Length':>8} | {'(ms)':>12} | {'':>12} | {'':>12} | {'(GB)':>10}")
    report_lines.append("-"*80)
    
    for r in pytorch_results:
        report_lines.append(
            f"{r['chunk_size']:>8} | {r['avg_time_ms']:>12.2f} | {r['tps']:>12,.0f} | "
            f"{r['chunks_per_sec']:>12.2f} | {r['memory_gb']:>10.2f}"
        )
    
    report_lines.append("")
    report_lines.append("")
    
    # vLLM Results
    if vllm_results:
        report_lines.append("="*80)
        report_lines.append("PART 2: VLLM OPTIMIZED RESULTS")
        report_lines.append("="*80)
        report_lines.append("")
        report_lines.append(f"{'Context':>8} | {'Batch':>6} | {'Time/Chunk':>12} | {'TPS':>12} | {'Chunks/Sec':>12} | {'Memory':>10}")
        report_lines.append(f"{'Length':>8} | {'Size':>6} | {'(ms)':>12} | {'':>12} | {'':>12} | {'(GB)':>10}")
        report_lines.append("-"*80)
        
        for r in vllm_results:
            report_lines.append(
                f"{r['chunk_size']:>8} | {r['batch_size']:>6} | {r['time_per_chunk_ms']:>12.2f} | "
                f"{r['tps']:>12,.0f} | {r['chunks_per_sec']:>12.2f} | {r['memory_gb']:>10.2f}"
            )
        
        report_lines.append("")
        report_lines.append("")
        
        # Speed comparison
        report_lines.append("="*80)
        report_lines.append("PYTORCH VS VLLM SPEEDUP")
        report_lines.append("="*80)
        report_lines.append("")
        report_lines.append(f"{'Context':>8} | {'Batch':>6} | {'PyTorch TPS':>15} | {'vLLM TPS':>15} | {'Speedup':>10}")
        report_lines.append(f"{'Length':>8} | {'Size':>6} | {'':>15} | {'':>15} | {'':>10}")
        report_lines.append("-"*80)
        
        for pytorch_r in pytorch_results:
            chunk_size = pytorch_r['chunk_size']
            pytorch_tps = pytorch_r['tps']
            
            # Find best vLLM result for this chunk size
            vllm_matches = [r for r in vllm_results if r['chunk_size'] == chunk_size]
            if vllm_matches:
                best_vllm = max(vllm_matches, key=lambda x: x['tps'])
                speedup = best_vllm['tps'] / pytorch_tps
                
                report_lines.append(
                    f"{chunk_size:>8} | {best_vllm['batch_size']:>6} | {pytorch_tps:>15,.0f} | "
                    f"{best_vllm['tps']:>15,.0f} | {speedup:>10.2f}x"
                )
        
        report_lines.append("")
        report_lines.append("")
    
    # 10T Token Projections
    report_lines.append("="*80)
    report_lines.append("10 TRILLION TOKEN PROCESSING PROJECTIONS")
    report_lines.append("="*80)
    report_lines.append("")
    
    total_tokens = 10_000_000_000_000  # 10 trillion
    
    # Find optimal config (highest TPS)
    if vllm_results:
        optimal = max(vllm_results, key=lambda x: x['tps'])
        optimal_type = "vLLM"
    else:
        optimal = max(pytorch_results, key=lambda x: x['tps'])
        optimal_type = "PyTorch"
    
    optimal_chunk_size = optimal['chunk_size']
    optimal_tps = optimal['tps']
    optimal_batch = optimal.get('batch_size', 1)
    
    seconds_per_gpu = total_tokens / optimal_tps
    hours_per_gpu = seconds_per_gpu / 3600
    days_per_gpu = hours_per_gpu / 24
    
    report_lines.append(f"Optimal Configuration:")
    report_lines.append(f"  Type: {optimal_type}")
    report_lines.append(f"  Chunk size: {optimal_chunk_size:,} tokens")
    report_lines.append(f"  Batch size: {optimal_batch}")
    report_lines.append(f"  Throughput: {optimal_tps:,.0f} TPS")
    report_lines.append("")
    report_lines.append(f"Time to process 10 trillion tokens:")
    report_lines.append(f"  1 H100:   {days_per_gpu:6.2f} days  ({hours_per_gpu:7.1f} hours)")
    report_lines.append(f"  4 H100s:  {days_per_gpu/4:6.2f} days  ({hours_per_gpu/4:7.1f} hours)")
    report_lines.append(f"  10 H100s: {days_per_gpu/10:6.2f} days  ({hours_per_gpu/10:7.1f} hours)")
    report_lines.append(f"  20 H100s: {days_per_gpu/20:6.2f} days  ({hours_per_gpu/20:7.1f} hours)")
    report_lines.append("")
    report_lines.append(f"Cost estimates (H100 @ $2/hour):")
    report_lines.append(f"  1 GPU:   ${hours_per_gpu * 2:,.0f}")
    report_lines.append(f"  4 GPUs:  ${(hours_per_gpu/4) * 2 * 4:,.0f}")
    report_lines.append(f"  10 GPUs: ${(hours_per_gpu/10) * 2 * 10:,.0f}")
    report_lines.append(f"  20 GPUs: ${(hours_per_gpu/20) * 2 * 20:,.0f}")
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("END OF REPORT")
    report_lines.append("="*80)
    
    # Write to file
    report_text = "\n".join(report_lines)
    with open(report_filename, 'w') as f:
        f.write(report_text)
    
    # Also log to console
    logger.info(report_text)
    logger.info("")
    logger.info(f"📄 Full report saved to: {report_filename}")
    logger.info(f"📄 Detailed log saved to: {log_filename}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Configuration
    model_path = "google/gemma-3-1b-it"
    context_lengths = [2000, 4000, 8000, 16000, 32000]
    batch_sizes = [1, 8, 16, 32, 64]  # For vLLM
    num_runs = 50  # Number of runs per benchmark
    num_runs_vllm = 20  # Fewer runs for vLLM (already batched)
    
    logger.info(f"Model: {model_path}")
    logger.info(f"Context lengths to test: {context_lengths}")
    logger.info(f"Batch sizes (vLLM): {batch_sizes}")
    logger.info(f"Runs per benchmark (PyTorch): {num_runs}")
    logger.info(f"Runs per benchmark (vLLM): {num_runs_vllm}")
    logger.info("")
    
    # Create test text
    test_text = create_test_text()
    logger.info(f"Test text length: {len(test_text.split())} words")
    logger.info("")
    
    # Load models
    tokenizer, model = load_model(model_path)
    
    # Run benchmarks
    pytorch_results = benchmark_pytorch(
        model, tokenizer, test_text, context_lengths, num_runs
    )
    
    vllm_results = benchmark_vllm(
        model_path, tokenizer, test_text, context_lengths, batch_sizes, num_runs_vllm
    )
    
    # Generate comprehensive report
    generate_report(pytorch_results, vllm_results, report_filename)
    
    logger.info("")
    logger.info("="*80)
    logger.info("BENCHMARK COMPLETE!")
    logger.info("="*80)
    logger.info(f"📄 Report: {report_filename}")
    logger.info(f"📄 Log:    {log_filename}")
    logger.info("")