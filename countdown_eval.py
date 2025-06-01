import json, re, math, random, ast, operator as op
import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    matches = list(re.finditer(r'<answer>(.*?)</answer>', solution_str, re.S))
    return matches[-1].group(1).strip() if matches else None

def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]
        
        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)
        
        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception as e:
        return None

def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    target = ground_truth['target']
    numbers = ground_truth['numbers']
    
    equation = solution_str
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0
    
    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return format_score
        
    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return format_score
            
        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except:
        if do_print:
            print(f"Error evaluating equation")
        return format_score 

EVAL_FILE   = "Countdown_eval.txt"
OUT_FILE    = "countdown_predictions.json"
CKPT_PATH   = "models/sft/epochs_2-batch_4-lr_1e-05-seed_42-curr_type_curriculum-scheduler_True-static_True.pt"
MODEL_NAME  = "Qwen/Qwen2.5-0.5B"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

MAX_NEW_TOKENS = 1024
TEMPERATURE     = 0

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def prompt_template(nums, tgt):
      return (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\n"
        f"User: Using the numbers {nums}, create an equation that equals "
        f"{tgt}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
        "Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, \n"
        "for example <answer> (1 + 2) / 3 </answer>.\n"
        "Assistant: Let me solve this step by step.")
          

def main():
    set_seed()

    tok   = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, sliding_window=None)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

    eval_set = load_dataset("json", data_files=EVAL_FILE, split="train")

    total_points = 0.0
    with open(OUT_FILE, "w", buffering=1) as fout:          # line-buffered
        for ex in tqdm(eval_set, desc="Generating"):
            nums, tgt = ex["num"], ex["target"]

            prompt_ids = tok(prompt_template(nums, tgt),
                             return_tensors="pt").to(DEVICE)
            
            gen_ids = model.generate(
                **prompt_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )[0][prompt_ids["input_ids"].shape[1]:]

            raw_out = tok.decode(gen_ids, skip_special_tokens=True)
            expr = extract_solution(raw_out) 

            print(raw_out)

            # --- Scoring (all parsing inside compute_score) -------------
            print("Extracted: " + str(expr))
            pts = compute_score(
                expr,
                {"target": tgt, "numbers": nums}
            )
            total_points += pts

            # --- JSON output (uses extract_solution for the expression) ---
            
            fout.write(json.dumps({
                "num": nums,
                "target": tgt,
                "response": expr
            }) + "\n")

    print(f"\nWrote {OUT_FILE}")
    print(f"Total rubric points: {total_points:.1f} / {len(eval_set)}")

if __name__ == "__main__":
    main()

