import torch
import ast
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import re
import os


print("Loading model...")
model_path = "Path to the model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="cuda:0" if torch.cuda.is_available() else "cpu"
)
model.eval()
print("Model loaded successfully on:", model.device)



def build_prompt(symptoms: str, ehr: str,fewshot_examples:list,ret_diseases,k: int) -> str:
    prompt= f"""
<|start_header_id|>system<|end_header_id|>
You are an expert medical coding assistant.
Given patient symptoms and clinical information, predict valid ranked ICD-9 diagnosis codes.
You are also provided with possible diseases. Use those as hints to predict the ICD-9 codes.
Follow all rules exactly and output only valid ICD-9 codes.
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Predict exactly {k} ICD-9 diagnosis codes for the following symptoms.

Rules:
- Standard ICD-9 codes are 3 digits (e.g., 250, 401).
- Codes starting with 'E' or 'V' may be 4 characters (e.g., E930, V451).
- Return ONLY a valid Python list of strings.
- Each code must be prefixed with its rank (e.g., "1:250", "2:401").
- DO NOT include disease names, explanations, or reasoning.
- DO NOT repeat this prompt or include extra words.
- Example format: ["1:250", "2:401", "3:V451"]"""

    for ex in fewshot_examples:
            prompt += f"""
                Symptoms: {ex['symptoms']}
                ehr: {ex['ehr']}
                Correct ICD-9 Codes: {ex['true_codes']}
            """


    prompt+=f"""Now predict for the new case:
    Symptoms:
    {symptoms}

    Electronic Health Record (EHR) of patient:
    {ehr}

    Retreived possible diseases (Use as hints):
    {ret_diseases}
    
    Return only the Python list:
    <|eot_id|>

    """.strip()
    return prompt


def extract_ranked_list(text: str):
    try:
        body = text.split("[", 1)[-1].split("]", 1)[0]
        parsed = ast.literal_eval("[" + body + "]")
        if isinstance(parsed, list):
            return [str(c).strip() for c in parsed]
    except Exception:
        pass
    return []


def extract_ranked_regex(text: str):
    return [
        f"{rank}:{code}"
        for rank, code in re.findall(
            r"\b(\d+)\s*:\s*([VE]?\d+(?:\.\d+)?)\b",
            text.upper()
        )
    ]


def extract_final_ranked_codes(text: str):
    codes = extract_ranked_list(text)
    if not codes:
        codes = extract_ranked_regex(text)
    return codes


def predict_icd9(symptoms: str, ehr: str,fewshot_examples,ret_diseases,k: int, max_new_tokens: int = 2048):
    prompt = build_prompt(symptoms, ehr,fewshot_examples,ret_diseases,k)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        )

    output_text = tokenizer.decode(
        generated[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()
    del generated, inputs  
    torch.cuda.empty_cache()  
    codes = extract_final_ranked_codes(output_text)
    return codes



if __name__ == "__main__":
    print("Starting inference...")

    df = pd.read_csv("Path to mimic test set")
    print("Total rows:", len(df))

    fewshot_examples = []
    sample = df.sample(n=2, random_state=42)

    for _, ex in sample.iterrows():
        fewshot_examples.append({
            "symptoms": str(ex["symptoms"]),
            "true_codes": [
                c.split(".")[0].upper().strip()
                for c in str(ex["short_codes"]).split(",")
                if c.strip()
            ],
            "ehr": str(ex['ehr'])
        })

    ranked_fewshot_examples = []
    for ex in fewshot_examples:
        ranked_fewshot_examples.append({
            "symptoms": ex["symptoms"],
            "true_codes": [f"{i+1}:{code}" for i, code in enumerate(ex["true_codes"])],
            "ehr": ex["ehr"]
        })

    fewshot_examples = ranked_fewshot_examples
    
    results = []
    for idx, row in enumerate(df.itertuples(), start=0):
        print(f"Processing row {idx}/{len(df)}")

        true_codes = [c.strip().upper() for c in row.short_codes.split(",") if c.strip()]
        ehr=row.ehr
        # number of disease codes to retreive [5,10,15]
        k = 15
        ret_diseases=row.predicted_diseases
        pred_codes = predict_icd9(row.symptoms,ehr,fewshot_examples,ret_diseases,k)
        print(pred_codes)
        results.append({
            "Symptoms": row.symptoms,
            "True_Codes": true_codes,
            "Predicted_Codes": pred_codes
        })
    results_df = pd.DataFrame(results)
    save_path = "Path to save the predictions"
    results_df.to_csv(save_path, index=False)

    print(f"Inference completed and saved to: {save_path}")


