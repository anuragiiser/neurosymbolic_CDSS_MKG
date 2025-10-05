mport csv
import ast
from neo4j import GraphDatabase

# Connect to the Neo4j database
uri = "bolt://localhost:7687"
username = "neo4j"
password = "sparse_kg_2_123"
driver = GraphDatabase.driver(uri, auth=(username, password))

# Function to query Neo4j based on a list of symptoms
def query_diseases_for_symptoms(symptom_list, true_length):

    query = """
     MATCH (s:Symptom)-[r:HAS_SYMPTOM]-(d:Disease)
     WHERE s.name IN $symptom_list OR any(synonym IN s.synonyms WHERE synonym IN $symptom_list)
     RETURN d.id AS disease_id, r.weight AS relevance
     ORDER BY relevance DESC
     LIMIT $true_length
    """

    with driver.session() as session:
        result = session.run(query, symptom_list=symptom_list, true_length=true_length)
        return {record["disease_id"] for record in result}

# Function to calculate precision, recall, and F1 score based on predicted and true disease codes
def calculate_f1(true_codes, predicted_codes):
    true_prefixes = {code[:3] for code in true_codes}
    pred_prefixes = {str(code)[:3] for code in predicted_codes}

    # True Positives (TP): Codes correctly predicted
    true_positives = len(true_prefixes & pred_prefixes)
    
    # False Positives (FP): Predicted codes that are not in true codes
    false_positives = len(pred_prefixes - true_prefixes)
    
    # False Negatives (FN): True codes that were not predicted
    false_negatives = len(true_prefixes - pred_prefixes)
    
    # Calculate Precision and Recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Calculate F1 Score
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1

# Load test samples
test_file_path = "symptoms_test.csv"
symptom_lists = []
true_label_sets = []

with open(test_file_path, newline='', encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # Parse the 'symptoms' column as a Python list
        symptom_list = ast.literal_eval(row["Symptoms"])
        symptom_lists.append(symptom_list)

        # Parse the 'correct_disease_codes' column as a list of disease codes
        true_diseases = row["short_codes"].split(",")
        true_label_sets.append(set(true_diseases))

# Open output CSV to save F1 scores for each sample
output_file_path = "hkg_fr11.csv"
with open(output_file_path, mode='w', newline='', encoding="utf-8") as outfile:
    fieldnames = ["symptoms", "true_disease_codes", "predicted_diseases", "precision", "recall", "f1_score"]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    # Generate predictions and calculate F1 scores
    f1_scores = []
    prec = []
    rec = []
    for symptoms, true_diseases in zip(symptom_lists, true_label_sets):
        true_length = 22
        # print(symptoms)
        predicted_diseases = query_diseases_for_symptoms(symptoms, true_length=true_length)
        # print(predicted_diseases)

        # Calculate precision, recall, and F1 score
        precision, recall, f1 = calculate_f1(true_diseases, predicted_diseases)

        f1_scores.append(f1)
        prec.append(precision)
        rec.append(recall)
        
        # Write the results for this sample to the CSV file
        writer.writerow({
            "symptoms": ", ".join(symptoms),
            "true_disease_codes": ", ".join(true_diseases),
            "predicted_diseases": ", ".join(map(str, predicted_diseases)),
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

# Calculate and print the Macro F1 score across all samples
macro_f1_score = sum(f1_scores) / len(f1_scores)
macro_prec = sum(prec) / len(prec)
macro_rec = sum(rec) / len(rec)
print(f"Macro Precision Score: {macro_prec:.4f}")
print(f"Macro Recall Score: {macro_rec:.4f}")
print(f"Macro F1 Score: {macro_f1_score:.4f}")
print(len(f1_scores))

# Close the Neo4j connection
driver.close()
