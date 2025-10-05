import os
import csv
import json
from collections import defaultdict

# Load ICD-9 CM codes into a dictionary
def load_icd9_codes(filepath):
    icd9_dict = {}
    with open(filepath, 'r') as f:
        for line in f:
            code, name = line.strip().split(maxsplit=1)
            icd9_dict[name.lower()] = code
    return icd9_dict

# Process each CSV and aggregate into a single JSON structure
def process_folder_to_json(folder_path, icd9_file, output_json_file):
    icd9_dict = load_icd9_codes(icd9_file)

    all_disease_data = []

    # Loop through files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            csv_file = os.path.join(folder_path, filename)
            
            # Extract the disease name from the filename (e.g., Bell's_palasy_symptom_count.csv)
            disease_name = filename.split('_symptom_counts.csv')[0].replace('_', ' ')
            disease_id = icd9_dict.get(disease_name.lower(), None)
            
            if not disease_id:
                print(f"Disease ID not found for {disease_name}")
                continue

            # Dictionary to track symptoms by CUI
            symptom_dict = defaultdict(lambda: {"synonyms": [], "positive_count": 0})
            
            # Read the CSV file and populate the symptom dictionary
            with open(csv_file, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    cui = row['cui']
                    symptom_name = row['symptom']
                    positive_count = int(row['positive_count'])
                    
                    # Add symptom as synonym for the same CUI
                    symptom_dict[cui]["synonyms"].append(symptom_name)
                    symptom_dict[cui]["positive_count"] += positive_count
            
            # Prepare the disease data
            disease_data = {
                "disease_name": disease_name,
                "disease_id": disease_id,
                "symptoms": []
            }
            
            # Convert the symptom dictionary to the required format
            for cui, data in symptom_dict.items():
                symptom_data = {
                    "symptom_cui": cui,
                    "synonyms": list(set(data["synonyms"])),  # Remove duplicates
                    "positive_count": data["positive_count"]
                }
                disease_data["symptoms"].append(symptom_data)
            
            # Add the disease data to the final JSON structure
            all_disease_data.append(disease_data)
    
    # Write the final JSON data to an output file
    with open(output_json_file, 'w') as jsonfile:
        json.dump(all_disease_data, jsonfile, indent=4)
    print(f"Consolidated JSON file created at {output_json_file}")

# Example usage
folder_path = "sparse_results_cui"
icd9_file = "icd9_cm_codes.txt"
output_json_file = "sparse_kg_data.json"

process_folder_to_json(folder_path, icd9_file, output_json_file)
