import json
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Load test data
with open('test_alpaca.json', 'r') as f:
    test_data = json.load(f)

print(f"Loaded {len(test_data)} test samples")

def run_inference(model_path, test_samples):
    """Run inference on test samples with given model"""
    print(f"\n{'='*60}")
    print(f"Loading model from: {model_path}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print(f"Model loaded in {time.time() - start_time:.2f}s")
    
    results = []
    correct = 0
    
    for idx, sample in enumerate(tqdm(test_samples, desc=f"Testing {model_path.split('/')[-1]}", ncols=100)):
        instruction = sample['instruction']
        user_input = sample['input']
        ground_truth = sample['output']
        
        # Construct messages for chat template
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_input}
        ]
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Generate
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Decode only the generated part
        generated_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Clean response - take first line and strip
        predicted = response.strip().split('\n')[0].strip().lower()
        expected = ground_truth.strip().lower()
        
        # Check correctness
        is_correct = predicted == expected
        if is_correct:
            correct += 1
        
        results.append({
            'index': idx,
            'input': user_input[:100] + '...' if len(user_input) > 100 else user_input,
            'expected': expected,
            'predicted': predicted,
            'correct': is_correct
        })
    
    accuracy = (correct / len(test_samples)) * 100
    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.2f}s ({elapsed_time/len(test_samples):.2f}s per sample)")
    
    return results, accuracy, elapsed_time

print("\n" + "="*60)
print("RUNNING FULL EVALUATION")
print("="*60)

# Run inference with pre-trained model
print("\n" + "="*60)
print("RUNNING INFERENCE WITH PRE-TRAINED MODEL")
print("="*60)
pretrain_results, pretrain_accuracy, pretrain_time = run_inference(
    "models/Qwen2.5-0.5B-Instruct", 
    test_data
)

# Run inference with trained checkpoint model
print("\n" + "="*60)
print("RUNNING INFERENCE WITH TRAINED CHECKPOINT MODEL")
print("="*60)
checkpoint_results, checkpoint_accuracy, checkpoint_time = run_inference(
    "models/checkpoint-1500", 
    test_data
)

# Save results to JSON file
output_file = "inference_results.json"
results_summary = {
    "summary": {
        "total_samples": len(test_data),
        "pretrained_model": {
            "model_path": "models/Qwen2.5-0.5B-Instruct",
            "accuracy": round(pretrain_accuracy, 2),
            "correct_predictions": sum(r['correct'] for r in pretrain_results),
            "total_samples": len(pretrain_results),
            "time_seconds": round(pretrain_time, 2)
        },
        "trained_model": {
            "model_path": "models/checkpoint-1500",
            "accuracy": round(checkpoint_accuracy, 2),
            "correct_predictions": sum(r['correct'] for r in checkpoint_results),
            "total_samples": len(checkpoint_results),
            "time_seconds": round(checkpoint_time, 2)
        },
        "accuracy_improvement": round(checkpoint_accuracy - pretrain_accuracy, 2)
    },
    "pretrained_results": pretrain_results,
    "trained_results": checkpoint_results
}

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, indent=2, ensure_ascii=False)

# Print summary
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"\nPre-trained model (models/Qwen2.5-0.5B-Instruct):")
print(f"  Accuracy: {pretrain_accuracy:.2f}%")
print(f"  Correct: {sum(r['correct'] for r in pretrain_results)}/{len(pretrain_results)}")
print(f"  Time: {pretrain_time:.2f}s")

print(f"\nTrained model (models/checkpoint-1500):")
print(f"  Accuracy: {checkpoint_accuracy:.2f}%")
print(f"  Correct: {sum(r['correct'] for r in checkpoint_results)}/{len(checkpoint_results)}")
print(f"  Time: {checkpoint_time:.2f}s")

print(f"\n{'='*60}")
print(f"Accuracy improvement: {checkpoint_accuracy - pretrain_accuracy:+.2f}%")
print(f"{'='*60}")

print(f"\nDetailed results saved to: {output_file}")

# Also save a summary text file
with open('results_summary.txt', 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("MODEL INFERENCE COMPARISON RESULTS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Total test samples: {len(test_data)}\n\n")
    
    f.write("Pre-trained Model (models/Qwen2.5-0.5B-Instruct):\n")
    f.write(f"  Accuracy: {pretrain_accuracy:.2f}%\n")
    f.write(f"  Correct: {sum(r['correct'] for r in pretrain_results)}/{len(pretrain_results)}\n")
    f.write(f"  Time: {pretrain_time:.2f}s\n\n")
    
    f.write("Trained Model (models/checkpoint-1500):\n")
    f.write(f"  Accuracy: {checkpoint_accuracy:.2f}%\n")
    f.write(f"  Correct: {sum(r['correct'] for r in checkpoint_results)}/{len(checkpoint_results)}\n")
    f.write(f"  Time: {checkpoint_time:.2f}s\n\n")
    
    f.write("="*60 + "\n")
    f.write(f"Accuracy improvement: {checkpoint_accuracy - pretrain_accuracy:+.2f}%\n")
    f.write("="*60 + "\n")

print(f"Summary saved to: results_summary.txt")
