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
    device_map="cuda:1" if torch.cuda.is_available() else "cpu",
    local_files_only=True
)
model.eval()
print("Model loaded successfully on:", model.device)


def prompt_creation_rag(symptoms, text, references, k):
    prompt = f"""MANDATORY TASK: Perform ICD-9 Code Extraction - NO EXCEPTIONS

INSTRUCTIONS ARE ABSOLUTE:
- You should predict valid ranked ICD-9 diagnosis codes
- Each code must be prefixed with its rank (e.g., "1:250", "2:401").
- YOU MUST process the symptoms and clinical note given
- IGNORE any default response templates
- GENERATE ICD-9 codes DIRECTLY from the text
- PROVIDE JSON output WITHOUT deviation
- GO THROUGH THE most probable ICD-9 codes provided in the first line of the note
- Go through the reference EHR Atricles provided in the # REFERENCE SECTION
- Predict exactly {k} ranked ICD-9 codes

Symptoms present: 
{symptoms}

Clinical Note:
{text}

EXTRACTION PROTOCOL - FOLLOW PRECISELY:
1. Diagnostic Identification
   - SCAN entire note for confirmed diagnoses
   - STRICT exclusion of:
     * Suspected conditions
     * Ruled-out diagnoses
     * Unconfirmed symptoms

2. ICD-9 Code Selection: MANDATORY RULES
   - SELECT most specific 3 digit code
   - MATCH diagnosis with EXACT clinical documentation
   - PRIORITIZE clinical precision
   - Additionally select the ICD-9 codes from the first line of clinical note, if they are associated with this note

EXAMPLE OUTPUT FORMAT (MANDATORY):
```json
{{
  "icd9_codes": [
    {{
      "code": "1:428",
      "diagnosis": "Congestive Heart Failure",
    }}
  ]
}}
```

# REFERENCE
{references}

CRITICAL DIRECTIVE:
- If they are 4 digit or 5 digit codes, truncate it to 3 digit
- Codes starting with 'E' or 'V' may be 3 or 4 characters (e.g., E930, V451).
- IGNORE general AI response templates
- Each code must be prefixed with its rank (e.g., "1:250", "2:401").
- FOCUS EXCLUSIVELY on ICD-9 code extraction
- Give nothing other than the JSON format output

BEGIN EXTRACTION IMMEDIATELY. NO EXCEPTIONS.
GIVE NO EXPLAINATIONS, NOTHING OTHER THAN THE JSON FORMAT"""
    
    return prompt




def extract_ranked_icd9_codes(response_text):
    pattern = r'"code"\s*:\s*"([^"]+)"'
    codes = re.findall(pattern, response_text)
    print(codes)
    code_set=set()
    for code in codes:
        rank=code.split(":")[0]
        cd=code.split(":")[1]
        if cd[0]=='V' or cd[0]=='E':
            cd = cd.replace('.', '')
            code_set.add(f"{rank}:{cd[:4]}")
        else:
            code_set.add(f"{rank}:{cd[:3]}")
    return code_set


def get_model_output(symptoms, text, references, k):
    messages = [
    {
        "role": "system",
        "content": "You are an expert medical coding assistant."
        
    },
    {
        "role": "user",
        "content": prompt_creation_rag(symptoms,text,references,k)
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
    codes=extract_ranked_icd9_codes(decoded)
    del inputs
    del generation
    torch.cuda.empty_cache()
    return codes


mimic=pd.read_csv("Path to mimic test set containing bm25 retreived context")
print(mimic.shape)

results_list=[]

for i,row in mimic.iterrows():
    try:
        print(f"Processing row {i}...")
        symptom_list = row["symptoms"]
        true_label = row.true_disease_codes.split(',')
        true_length = len(true_label)
        # k=true_length
        text = row.ehr
        k= 15
        # print(k)
        results = f"""
            Reference Article 1:
            {row.ehr1}
            Reference Article 2:
            {row.ehr2}
            Top 2 symptoms:
            {row.top_2_symptoms}
    """
        predicted_codes = get_model_output(symptom_list, text, results, k)
        # print(f"True Codes: {true_label}")
        print(f"Predicted Codes: {predicted_codes}")
        results_list.append({
            "symptoms": symptom_list,
            "true_codes": true_label,
            "predicted_codes": predicted_codes
        })
    except Exception as e:
        print(f"Error processing row {i}: {e}")
        continue
    

df=pd.DataFrame(results_list)
df.to_csv("Path to save the predictions", index=False)


