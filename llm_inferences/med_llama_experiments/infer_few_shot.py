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
    load_in_4bit=True,
    device_map="cuda:1" if torch.cuda.is_available() else "cpu",
    local_files_only=True,
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


def build_prompt(symptoms: str,ehr,fewshot_examples,k:int) -> str:
    prompt=f"""
    Given patient symptoms and clinical information, predict valid ranked ICD-9 diagnosis codes.
    Follow all rules exactly and output only valid ICD-9 codes.
    Predict exactly {k} ICD-9 diagnosis codes for the following symptoms.

    Rules:
    - Standard ICD-9 codes are 3 digits (e.g., 250, 401) and codes starting with 'E' or 'V'.
    - Codes starting with 'E' or 'V' may be 4 characters (e.g., E930, V451).
    - Return ONLY a valid Python list of strings.
    - Each code must be prefixed with its rank (e.g., "1:250", "2:401").
    - DO NOT include disease names, explanations, or reasoning.
    - DO NOT repeat this prompt or include extra words.
    - Example format: ["1:250", "2:401", "3:V451"]
    """

    prompt += f"""
    Now predict for the case:
        Symptoms:
        {symptoms}

        Electronic Health Record (EHR) of patient:
        {ehr}

        Below are some few shot examples:
    """.strip()

    for ex in fewshot_examples:
        prompt += f"""
            Symptoms: {ex['symptoms']}
            ehr: {ex['ehr']}
            Correct ICD-9 Codes: {ex['true_codes']}
        """

    prompt += f"""
        Return only the Python list:
    """.strip()    
    return prompt



def predict_icd9(symptoms: str, ehr,few_shot,k: int=15):
    messages = [
    {
        "role": "system",
        "content": "You are an expert medical coding assistant."
        
    },
    {
        "role": "user",
        "content": build_prompt(symptoms,ehr,few_shot,k)
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



df = pd.read_csv("Path to mimic test set")
print("Total rows:", len(df))

fewshot_examples = []
sample = df.sample(n=2, random_state=42)

for _, ex in sample.iterrows():
    fewshot_examples.append({
        "symptoms": str(ex["symptoms"]),
        "true_codes": [
            c.split(".")[0].upper().strip()
            for c in str(ex["true_disease_codes"]).split(",")
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

for i, row in df.iterrows():
    try:
        print(f"Processing case {i+1}/{len(df)}")
        symptoms = str(row["symptoms"])
        true_codes = [c.strip().upper() for c in str(row["true_disease_codes"]).split(",") if c.strip()]
        ehr=row['ehr']
        k=15
        pred_codes=predict_icd9(symptoms,ehr,fewshot_examples,k=15)
        print(pred_codes)
        results.append({
            "symptoms": symptoms,
            "true_codes": true_codes,
            "predicted_codes": pred_codes
        })
    except Exception as e:
        print(f"Error processing case {i+1}: {e}")
        continue

out_df = pd.DataFrame(results)
output_path = "Path to save the predictions"
out_df.to_csv(output_path, index=False)
print("Saved:", output_path)
