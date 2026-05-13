import tqdm
import torch
import ast
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import re


print("Loading model...")
model_path = "Path to the model"

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    local_files_only=True
)
model.eval()
print("Model loaded successfully on:", model.device)


import ast
import re


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


def build_prompt(symptoms: str,ehr,ret_diseases,k:int) -> str:
    return f"""
    Given patient symptoms and clinical information, predict valid ranked ICD-9 diagnosis codes concisely.
    Follow all rules exactly and output only valid ICD-9 codes.
    You are also provided with possible diseases. Use those as hints to predict the ICD-9 codes.
    Predict exactly {k} ICD-9 diagnosis codes for the following symptoms.

    Rules:
    - Standard ICD-9 codes are 3 digits (e.g., 250, 401) and code starting with 'E' or 'V'.
    - Codes starting with 'E' or 'V' may be 4 characters (e.g., E930, V451).
    - Return ONLY a valid Python list of strings.
    - Each code must be prefixed with its rank (e.g., "1:250", "2:401").
    - DO NOT include disease names, explanations, or reasoning.
    - DO NOT repeat this prompt or include extra words.
    - Example format: ["1:250", "2:401", "3:V451"]

    Symptoms:
    {symptoms}

    Electronic Health Record (EHR) of patient:
    {ehr}

    Retreived possible diseases (Use as hints):
    {ret_diseases}

    Return only the Python list:
    """.strip()


def predict_icd9(symptoms: str, ehr,ret_diseases,k: int=15):
    messages = [
    {
        "role": "system",
        "content": "You are an expert medical coding assistant."
        
    },
    {
        "role": "user",
        "content": build_prompt(symptoms,ehr,ret_diseases,15)
    }
    ]
    inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
        generation = generation[0][input_len:]

    decoded = tokenizer.decode(generation, skip_special_tokens=True)
    codes=extract_final_ranked_codes(decoded)
    del inputs
    del generation
    torch.cuda.empty_cache()
    return codes



if __name__ == "__main__":
    print("Starting inference...")

    df = pd.read_csv("Path to mimic test set")
    print("Total rows:", len(df))

    results = []
    for idx, row in enumerate(df.itertuples(), start=0):
        print(f"Processing row {idx}/{len(df)}")
        true_codes = [c.strip().upper() for c in row.short_codes.split(",") if c.strip()]
        ret_diseases=row.predicted_diseases
        pred_codes = predict_icd9(row.symptoms,row.ehr,ret_diseases,15)
        print("Predicted Codes:", pred_codes)
        results.append({
            "Symptoms": row.symptoms,
            "True_Codes": true_codes,
            "Predicted_Codes": pred_codes
        })

    results_df = pd.DataFrame(results)
    save_path = "Path to save the predictions"
    results_df.to_csv(save_path, index=False)

    print(f"Inference completed and saved to: {save_path}")

