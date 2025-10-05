import pandas as pd
import os
import json
import pandas as pd

def prompt_creation(clinical_note):
    prompt = f"""MANDATORY TASK: Perform ICD-9 Code Extraction - NO EXCEPTIONS

INSTRUCTIONS ARE ABSOLUTE:
- YOU MUST process this entire clinical note
- IGNORE any default response templates
- GENERATE ICD-9 codes DIRECTLY from the text
- PROVIDE JSON output WITHOUT deviation
- GO THROUGH THE most probable ICD-9 codes provided in the first line of the note

CLINICAL NOTE:
{clinical_note}

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
      "code": "428",
      "diagnosis": "Congestive Heart Failure",
    }}
  ]
}}
```

CRITICAL DIRECTIVE:
- If they are 4 digit or 5 digit codes, truncate it to 3 digit
- IGNORE general AI response templates
- FOCUS EXCLUSIVELY on ICD-9 code extraction
- Give nothing other than the JSON format output

BEGIN EXTRACTION IMMEDIATELY. NO EXCEPTIONS.
GIVE NO EXPLAINATIONS, NOTHING OTHER THAN THE JSON FORMAT"""

    return prompt


def prompt_creation_rag(clinical_note, references):
    prompt = f"""MANDATORY TASK: Perform ICD-9 Code Extraction - NO EXCEPTIONS

INSTRUCTIONS ARE ABSOLUTE:
- YOU MUST process this entire clinical note
- IGNORE any default response templates
- GENERATE ICD-9 codes DIRECTLY from the text
- PROVIDE JSON output WITHOUT deviation
- GO THROUGH THE most probable ICD-9 codes provided in the first line of the note
- Go through the reference PUBMed Atricles provided in the # REFERENCE SECTION

CLINICAL NOTE:
{clinical_note}

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
      "code": "428",
      "diagnosis": "Congestive Heart Failure",
    }}
  ]
}}
```

# REFERENCE
{references}

CRITICAL DIRECTIVE:
- If they are 4 digit or 5 digit codes, truncate it to 3 digit
- IGNORE general AI response templates
- FOCUS EXCLUSIVELY on ICD-9 code extraction
- Give nothing other than the JSON format output

BEGIN EXTRACTION IMMEDIATELY. NO EXCEPTIONS.
GIVE NO EXPLAINATIONS, NOTHING OTHER THAN THE JSON FORMAT"""

    return prompt





import json
import re

def extract_json_from_text(text):
    """
    Extract JSON content from a text that might include markdown code blocks.

    Args:
        text (str): Input text potentially containing JSON

    Returns:
        dict: Extracted and parsed JSON data
    """
    # Use regex to find content between triple backticks
    code_block_match = re.search(r'```\n*({.*?})\n*```', text, re.DOTALL)

    if code_block_match:
        json_str = code_block_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("Error: Could not parse JSON within code block")
            return None

    # If no code block found, try direct JSON parsing

    try:
        return json.loads(text.strip('` \n').replace('json\n', '', 1))
    except json.JSONDecodeError:
        print("Error: Could not parse input as JSON")
        return None



def preprocess_icd9_codes(input_text, id, output_dir = 'infer_llm_mimic_4'):
    """
    Preprocess ICD-9 codes with validation and cleaning.

    Args:
        input_text (str): Input text containing ICD-9 codes

    Returns:
        dict: Processed and validated ICD-9 codes
    """
    # First, extract JSON from the input text
    input_data = extract_json_from_text(input_text)

    if not input_data or 'icd9_codes' not in input_data:
        print("Error: No valid ICD-9 codes found")
        return {"icd9_codes": []}

    processed_codes = []

    for entry in input_data['icd9_codes']:
        # Validate code format (3 or 4 digit numeric)
        code = str(entry.get('code', '')).strip()
        if not re.match(r'^\d{3,4}$', code):
            print(f"Warning: Invalid code format - {code}")
            continue

        # Clean and standardize diagnosis
        diagnosis = entry.get('diagnosis', '').strip()
        diagnosis = re.sub(r'\s+', ' ', diagnosis)

        processed_codes.append({
            'code': code,
            'diagnosis': diagnosis
        })

    processed_data = {'icd9_codes': processed_codes}

    # Save processed data to JSON file
    with open(f'{output_dir}/icd9_codes_processed_{id}.json', 'w') as f:
        json.dump(processed_data, f, indent=2)

    return processed_data


from neo4j import GraphDatabase
from transformers import pipeline
import pandas as pd

# Connect to Neo4j
class Neo4jHandler:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query_kg(self, symptoms, true_length):
        """
        Query the KG for diseases related to the given symptoms.
        """
        query = """
        MATCH (s:Symptom)-[r:HAS_SYMPTOM]-(d:Disease)
    WHERE s.name IN $symptoms OR any(synonym IN s.synonyms WHERE synonym IN $symptoms)
    RETURN d.id AS disease, r.weight AS weight
    ORDER BY weight DESC
    LIMIT $true_length
        """
        with self.driver.session() as session:
            result = session.run(query, symptoms=symptoms, true_length=true_length)
            return [{"disease": record["disease"], "weight": record["weight"]} for record in result]

neo4j_handler = Neo4jHandler("bolt://localhost:8687", "neo4j", "neo4j_pass5")



"""## Get predictions with RAG
1. USING MIMIC : Pass the query expanded symptoms as well as the codes
2. USING THE PUBMED KG : Pass the output codes

## 1. MIMIC Approach
"""

#  pip install py2neo dotenv

import os
from py2neo import Graph
from mistralai.client import MistralClient
# from mistralai.models.chat_completion import ChatMessage
from neo4j import GraphDatabase

import json


uri = "neo4j://localhost:8687"
auth = ("neo4j", "neo4j_pass5")

driver = GraphDatabase.driver(uri, auth=auth)
driver.verify_connectivity()

"""#### For checking the matching symptoms"""

def escape_special_chars(query):
    return query.replace("'", "\\'").replace("/", "\\/")

def search_symptoms(query_strings,limit = 5):
    all_symptoms = []
    for query_string in query_strings:
        escaped_query = escape_special_chars(query_string)  # Escape the query string
        with driver.session() as session:
            result = session.run(
                f"CALL db.index.fulltext.queryNodes('symptomIndex', '{escaped_query}') "
                "YIELD node, score "
                "RETURN node.name AS symptom, score "
                "ORDER BY score DESC "
                f"LIMIT {limit}"
            )
            symptoms = [record['symptom'] for record in result]
            all_symptoms.extend(symptoms)
    return all_symptoms

# Example usage
search_queries = ['bilateral upper extremity ecchymosses', 'hypotension', 'leukocytosis', 'afib with RVR', 'acute renal failure', 'anemia', 'upper extremity ecchymoses', 'coagulopathy', 'thrombocytopenia', 'elevated white count', 'cryptogenic cirrhosis', 'chronic diastolic CHF', 'GI bleed', 'hypothyroid']
search_results = search_symptoms(search_queries)
print(search_results)

def get_prioritized_relationships(symptom_names, weightage=5, limit=10):
    """
    Retrieves a list of diseases associated with the given symptom names, ordered by the maximum weight of the associations.

    Parameters:
    symptom_names (list): A list of symptom names to search for.
    weightage (int, optional): The minimum weight threshold for the associations. Defaults to 5.
    limit (int, optional): The maximum number of results to return. Defaults to 20.

    Returns:
    tuple: A tuple containing:
        - codes (list): A list of disease names ordered by maximum association weight.
        - all_info (dict): A dictionary mapping disease names to a list containing the disease name and the maximum association weight.
    """
    with driver.session() as session:
        result = session.run(f"""
            MATCH (s:Symptom)-[r:ASSOCIATED_WITH]->(d:Disease)
            WHERE s.name IN $symptoms AND r.weight >= {weightage}
            WITH d.title AS disease_name, collect(s.name) AS symptoms, max(r.weight) AS max_weight
            RETURN disease_name, symptoms, max_weight
            ORDER BY max_weight DESC
            LIMIT {limit}
        """, symptoms=symptom_names)
        codes = []
        all_info = {}
        for record in result:
            codes.append(record['disease_name'])
            all_info[record['disease_name']] = [record['disease_name'], record['max_weight']]
        return codes, all_info

"""## 2. PUBMEDKG Approach"""

def get_predictions_with_rag(input_text, symptoms, true_length, threshold=0.8,  model_only = False ):
    """
    Get predictions from the model and refine them using the knowledge graph.
    """
    # Query the KG for related diseases
    if model_only :

        augmented_input = f"{input_text}"
    else :

        kg_results = neo4j_handler.query_kg(symptoms, true_length)
        augmented_input = f"{kg_results}\n{input_text}"




    # Tokenize and predict using the model
    tokenized_input = tokenizer(
        augmented_input,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding='max_length'
    )
    tokenized_input = {k: v.to(device) for k, v in tokenized_input.items()}
    output = model(**tokenized_input)
    predictions = torch.sigmoid(output.logits)
    predicted_labels = [model.config.id2label[_id] for _id in (predictions > threshold).nonzero()[:, 1].tolist()]


    return predicted_labels

def get_predictions(input_text, threshold = 0.8) :
    """give the EHR/Symptoms as input, and
    get the disease codes (matching in the classes under consideration)
    as the output"""

    tokenized_input =  tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,  # BERT's maximum sequence length
        padding='max_length'
    )
    tokenized_input = {k: v.to(device) for k, v in tokenized_input.items()}
    output = model(**tokenized_input)
    predictions = torch.sigmoid(output.logits)
    predicted_labels = [model.config.id2label[_id] for _id in (predictions > threshold).nonzero()[:, 1].tolist()]
    classes = str('403 486 582 585 425 276 710 724 458 287 285 275 583 558 327 228 338 789 790 V451 531 410 414 725 191 331 530 411 482 272 305 194 197 255 424 584 998 682 511 599 428 349 401 V100 V453 V586 041 251 E932 V300 V053 V290 571 070 250 570 572 286 518 038 280 263 995 303 244 112 881 903 955 E956 745 762 441 496 447 440 997 274 427 V104 V101 V120 V090 569 560 491 V458 433 436 493 996 416 V310 765 769 774 770 747 776 772 362 198 V103 746 766 V293 853 780 E888 730 357 430 293 443 V158 396 365 135 311 E935 721 214 437 242 600 189 304 711 800 E814 873 781 378 951 767 431 294 042 V141 V071 764 775 969 295 E950 266 779 355 553 965 E850 E853 426 804 E916 202 V502 398 707 348 787 564 V428 238 300 788 332 V107 V433 E879 861 423 E966 200 555 771 270 335 723 079 851 807 864 865 860 413 782 V108 507 512 752 162 783 778 333 785 136 799 E931 157 574 568 E878 722 719 V125 296 478 V170 805 596 E880 822 733 578 459 438 008 V098 185 967 225 V457 389 412 593 345 201 515 E933 278 492 715 415 V105 535 608 E870 V058 513 709 E821 V173 824 911 913 E812 576 203 281 580 V450 216 V340 579 693 351 088 714 E849 307 421 786 E942 959 E928 588 364 V642 V025 252 283 784 611 622 289 446 729 V498 V456 795 E854 V667 155 V130 882 852 957 E815 466 792 434 342 153 E934 481 910 456 453 867 273 532 806 V422 V541 556 394 444 924 E960 514 763 218 359 340 999 451 324 E939 537 737 455 E884 V427 591 592 577 557 575 356 368 552 500 750 253 292 E937 211 288 773 314 V652 432 379 435 E930 199 V641 494 966 758 E855 741 918 V436 078 562 820 801 839 E881 V584 731 E885 812 156 567 696 501 712 V707 215 754 753 508 876 720 V442 871 958 802 847 397 196 346 E968 510 404 360 376 370 V026 904 928 821 823 150 573 850 V497 E938 V533 V556 728 870 V874 V153 V644 V600 521 301 164 054 344 464 442 V150 282 V08 891 808 866 902 117 484 760 V048 691 519 528 320 369 685 V625 794 793 318 V441 761 936 E915 457 395 053 V113 V632 386 623 290 204 271 E819 811 813 884 E813 751 366 297 V440 473 E910 V420 057 536 152 970 485 235 372 E882 127 160 170 V880 595 909 V443 490 343 319 130 698 E823 246 854 868 872 982 151 V853 980 E980 291 517 268 487 E866 796 V452 036 354 648 701 V063 V038 227 614 533 736 942 E924 240 921 V454 977 759 768 923 E816 681 138 358 950 922 205 990 009 619 417 279 257 E860 755 991 E957 241 810 920 V461 V127 261 429 550 874 756 935 831 718 962 E858 803 480 674 277 880 879 377 529 047 083 835 462 336 E947 V160 420 317 454 E883 840 V550 960 586 933 597 350 E911 742 V614 298 V551 620 716 V462 V180 706 565 452 825 322 154 040 110 605 607 461 704 713 945 052 948 323 325 934 516 039 975 971 994 666 V111 907 E929 566 603 405 049 237 V161 V553 262 743 422 337 625 757 527 309 815 V163 402 869 E912 188 590 V852 V446 E852 886 E919 183 862 875 877 890 E944 E936 V444 598 V552 226 E818 617 E958 V123 748 968 V298 465 972 E826 905 E969 744 E829 V301 388 V146 V151 887 375 334 E848 E918 284 E876 260 987 E890 834 522 692 V588 310 863 E834 192 035 V174 171 738 220 477 212 172 V548 726 526 V099 777 749 E922 952 V320 901 542 449 V011 963 E822 524 V052 V539 144 445 321 380 604 383 587 137 845 695 V496 180 618 V102 540 525 916 174 V628 892 816 V171 520 708 176 791 V854 E906 V714 V554 V435 883 927 V434 007 581 V202 140 642 644 654 V270 V252 193 V838 V555 139 V195 V068 601 826 694 626 956 245 919 299 727 684 647 E941 V850 665 391 308 633 639 V230 V061 223 269 V183 046 534 361 673 643 986 005 034 382 239 232 V169 E901 908 634 836 616 E917 734 V698 133 E887 V445 V155 E949 142 E987 236 470 463 E940 229 448 702 182 E825 V851 814 V881 259 906 161 E891 830 E953 195 093 472 914 E988 930 543 686 900 075 705 939 381 V311 V168 018 004 917 483 656 641 217 V291 V164 E943 134 635 659 E920 506 E869 111 096 094 123 158 141 243 690 097 632 989 964 027 V596 373 V017 254 932 187 353 669 V504 602 843 912 374 983 E864 031 210 114 646 077 V018 670 615 V638 V135 938 V580 680 878 E965 471 652 663 658 V272 213 032 148 V643 V148 V062 E989 E927 131 233 V040 V066 125 V503 V581 V292 V192 700 703 209 V029 208 697 E871 184 015 146 V140 V154 992 249 149 V142 844 175 V542 363 V152 V106 V688 V265 012 885 E955 V530 385 V124 V741 390 474 627 817 230 E817 V198 E862 258 V463 735 V024 V640 976 E861 V765 V023 V626 E828 V188 341 V560 798 V448 893 495 084 523 V653 953 V549 V095 V182 621 475 V425 058 306 V165 551 E831 V136 V109 256 219 221 961 985 828 671 E820 897 V840 926 V421 048 594 896 082 E986 541 145 267 683 V097 732 265 011 E801 V185 664 V620 E840 V166 V468 629 115 V587 E908 120 V708 098 V469 V694 E824 E970 121 838 832 460 013 V239 944 V189 946 118 326 E945 645 352 159 E967 V618 147 V908 941 312 624 V186 V145 661 010 E865 091 E886 649 E905 E962 V612 E959 502 V438 V222 163 947 V162 E946 V716 315 367 V540 846 717 V561 V175 842 V138 V703 V583 841 672 062 488 347 339 E841 086 V400 E985 655 974 V289 V604 V074 V728 371 190 V126 090 143 943 V611 V331 085 V172 E835 668 740 V167 V558 E851 E811 V430 837 V072 V431 302 E923 V110 E900 V562 E963 E964 V118 V624 E800 988 833 023 V020 021 003 V660 E806 313 E954 V860 660 V449 231 V602 186 E863 E874 V721 V181 651 033 V654 E804 330 610 384 E838 E001 973 819 014 132 E899 925 207 V861 E002 E030 E000 894 E873 E999 E976 E003 V016 E805 045 V610 V078 V510 E029 848 E006 V403 122 V536 E013 E019 173 E913 677 E008 V568 V143 V091 V872 066 V601 116 V882 V065 538 V655 316 E007 E016 E921 V902 206 V254 099 V489 V870 E977 628 V250 E982 V486 539 V073 937 V812 030 V271 589 V672 V671 E926 E925 E857 V537 954 E827 657 V910 V789 V037 E975 V045 V848 393 V426 179 387 V903 E856 V901 915').split(' ')

    predicted_labels = list(set(classes).intersection(set(predicted_labels)))

    return predicted_labels

def create_binary_matrix(labels, classes):
    """
    Creates a binary matrix from a list of labels and a list of classes.

    Parameters:
    labels (list): A list of lists, where each inner list contains the labels for a single data point.
    classes (list): A list of class names.

    Returns:
    numpy.ndarray: A binary matrix where each row represents a data point and each column represents a class.
    """
    binary_matrix = np.zeros((len(labels), len(classes)), dtype=int)
    for i, label_list in enumerate(labels):
        for label in label_list:
            if label in classes:
                idx = classes.index(label)
                binary_matrix[i, idx] = 1
    return binary_matrix

# !pip install scikit-learn

from sklearn.metrics import roc_auc_score
import numpy as np

def roc_auc(probs, labels, multilabel=False, average='macro', multi_class='ovo'):
    if isinstance(labels, list):
        labels = np.array(labels, dtype=int)
    else:
        labels = labels.astype(int)

    # Filter relevant columns if multilabel is True
    y_score = probs
    if multilabel:
        # Identify classes with at least one positive label
        present_classes = np.any(labels == 1, axis=0)
        labels = labels[:, present_classes]
        y_score = np.array(probs)[:, present_classes]

    # Calculate ROC AUC score
    roc_auc = roc_auc_score(y_true=labels, y_score=y_score, average=average, multi_class=multi_class)

    return {
        'roc_auc': roc_auc
    }

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

# from collections import Counter

# my_list = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
# counts = Counter(my_list)
# print(counts)

import csv, ast
import pandas as pd
raw_test_df = pd.read_csv('symptoms_test.csv')
raw_test_df = raw_test_df.drop('Unnamed: 0',axis =1)

input_ =  str(raw_test_df.iloc[0].Symptoms)+ '\n' + str(raw_test_df.iloc[0].text)

input_

from tqdm import tqdm

expanded_query = search_symptoms(['fever'], limit = 2)
expanded_query

# neo4j_handler.query_kg(['fever', 'feverish'], true_length)

# import os

# os.makedirs('infer_llm_22')

import weaviate
from weaviate.connect import ConnectionParams

def mimic_train_datasets(client, text, top_n=2) :
    # Get the collection
    collection = client.collections.get("MedicalRecords_v3")

    # Perform a BM25 search
    response = collection.query.bm25(
        query=f"{text}",
        limit=top_n
    )

    results = []

    # Print results
    for item in response.objects:
        symptoms = item.properties['symptoms']
        codes = list(item.properties['short_codes'].split(','))
        icd_9_codes = [i[:3] for i in codes]
        text = item.properties['text']
        results.append((symptoms, icd_9_codes, text))
    return results



"""Testing the LLama 70b Model.

Case 1 : Base Model on MIMIC IV symptoms + notes

Case 2 : Base Model + KG  on MIMIC Symptoms + notes

Case 3 : Fine tuned model (smaller) on Mimic symptoms + notes

Case 4 : Base Model + KG on MIMIC Symptoms + notes
"""

mimic_4_df = pd.read_csv('mimic-iv-preprocessed-icd-symptoms.csv')
mimic_4_df.head()



outputs = []

"""## mimic -4 symptoms + ehr : llm"""

for i, j in tqdm(mimic_4_df.iterrows(), total=len(mimic_4_df)):
    symptom_list = ast.literal_eval(j["Symptoms"])
    true_label = j.SHORT_CODES.split(',')
    true_length = len(true_label)
    # kg_results = neo4j_handler.query_kg(symptom_list, 22)
    #disease_codes = [i['disease'][:3] for i in kg_results]

    input_ =  '\n' +  str(j.Symptoms)+ '\n'  + str(j.text) #+ str(disease_codes)
    output = get_icd9_codes_mistral(prompt_creation(input_), output_dir = '')
    outputs.append(output)

    processed_data = preprocess_icd9_codes(output, j.note_id)

    if i==500:
        break

"""## mimic 4 symptoms + ehr + PMKG : llm"""

for i, j in tqdm(mimic_4_df.iterrows(), total=len(mimic_4_df)):

    symptom_list = ast.literal_eval(j["Symptoms"])
    true_label = j.SHORT_CODES.split(',')
    true_length = len(true_label)
    kg_results = neo4j_handler.query_kg(symptom_list, 22)
    disease_codes = [i['disease'][:3] for i in kg_results]

    input_ = str(disease_codes) + '\n' +  str(j.Symptoms)+ '\n'  + str(j.text) #+ str(disease_codes)
    output = get_icd9_codes_mistral(prompt_creation(input_))
    outputs.append(output)

    processed_data = preprocess_icd9_codes(output, j.note_id, output_dir = 'infer_llm_mimic_4_kg')

    if i==500:
        break



import os
os.makedirs('infer_llm_mimic_4_kg')

# os.listdir('infer_llm_mimic_4')

mimic_4_df['SHORT_CODES'] = mimic_4_df['SHORT_CODES'].apply(lambda x : x.split(','))

mimic_4_df  = mimic_4_df.drop('Unnamed: 0',axis =1)

true_labels = []
pred_labels = []

for i in os.listdir('infer_llm_mimic_4_kg'):
    with open(f'infer_llm_mimic_4_kg/{i}', 'r') as f:
        note_id = i.split('icd9_codes_processed_')[1].split('.json')[0]

        data = json.loads(f.read())
        pred_codes = [i['code'] for i in data['icd9_codes']]


    # print(list(mimic_4_df[mimic_4_df.note_id == note_id].SHORT_CODES)[0])
    true_codes = list(mimic_4_df[mimic_4_df.note_id == note_id].SHORT_CODES)[0].split(',')

    true_labels.append(true_codes)
    pred_labels.append(pred_codes)



len(true_labels), len(pred_labels)

f1_scores , prec, rec = [], [], []
for true_label,pred_label in zip(true_labels, pred_labels) :
    precision, recall, f1 = calculate_f1(true_label, pred_label)
    f1_scores.append(f1)
    prec.append(precision)
    rec.append(recall)

macro_f1_score = sum(f1_scores) / len(f1_scores)
macro_prec = sum(prec) / len(prec)
macro_rec = sum(rec) / len(rec)
print(f"Macro Precision Score: {macro_prec:.4f}")
print(f"Macro Recall Score: {macro_rec:.4f}")
print(f"Macro F1 Score: {macro_f1_score:.4f}")

"""## mimic-4 (symptoms + EHR) + rag (mimic-3-train) ->  llama70b



"""

os.makedirs('infer_llm_mimic_rag_5')

client = weaviate.connect_to_local(
        port=9000,      # Custom HTTP port
        grpc_port=9001  # Custom gRPC port
    )

# Verify connection
print(f"Client is ready: {client.is_ready()}")


for i, j in tqdm(mimic_4_df.iterrows(), total=len(mimic_4_df)):
    if i<210:
        continue
    symptom_list = ast.literal_eval(j["Symptoms"])
    true_label = j.SHORT_CODES.split(',')
    true_length = len(true_label)
    text = j.TEXT
    # kg_results = neo4j_handler.query_kg(symptom_list, 22)
    #disease_codes = [i['disease'][:3] for i in kg_results]

    results  = mimic_train_datasets(client,text, top_n=5)

    input_ =  '\n' +  str(j.Symptoms)+ '\n'  + str(j.text) #+ str(disease_codes)
    output = get_icd9_codes_mistral(prompt_creation_rag(input_, results))

    processed_data = preprocess_icd9_codes(output, j.note_id, output_dir='infer_llm_mimic_rag_5')

    if i==500:
        break

client.close()

true_labels = []
pred_labels = []

for i in os.listdir('infer_llm_mimic_rag_5'):
    with open(f'infer_llm_mimic_rag_5/{i}', 'r') as f:
        note_id = i.split('icd9_codes_processed_')[1].split('.json')[0]

        data = json.loads(f.read())
        pred_codes = [i['code'] for i in data['icd9_codes']]


    # print(list(mimic_4_df[mimic_4_df.note_id == note_id].SHORT_CODES)[0])
    true_codes = list(mimic_4_df[mimic_4_df.note_id == note_id].SHORT_CODES)[0].split(',')

    true_labels.append(true_codes)
    pred_labels.append(pred_codes)

len(true_labels), len(pred_labels)

f1_scores , prec, rec = [], [], []
for true_label,pred_label in zip(true_labels, pred_labels) :
    precision, recall, f1 = calculate_f1(true_label, pred_label)
    f1_scores.append(f1)
    prec.append(precision)
    rec.append(recall)

macro_f1_score = sum(f1_scores) / len(f1_scores)
macro_prec = sum(prec) / len(prec)
macro_rec = sum(rec) / len(rec)
print(f"Macro Precision Score: {macro_prec:.4f}")
print(f"Macro Recall Score: {macro_rec:.4f}")
print(f"Macro F1 Score: {macro_f1_score:.4f}")

"""## mimic-4  (symptoms + EHR) + rag (mimic-3-train) + llama70b + KB


"""

os.makedirs('infer_llm_mimic_rag_kg_5')

client = weaviate.connect_to_local(
        port=9000,      # Custom HTTP port
        grpc_port=9001  # Custom gRPC port
    )

# Verify connection
print(f"Client is ready: {client.is_ready()}")


for i, j in tqdm(mimic_4_df.iterrows(), total=len(mimic_4_df)):
    if i<345:
        continue
    symptom_list = ast.literal_eval(j["Symptoms"])
    true_label = j.SHORT_CODES.split(',')
    true_length = len(true_label)
    text = j.TEXT
    kg_results = neo4j_handler.query_kg(symptom_list, 22)
    disease_codes = [i['disease'][:3] for i in kg_results]

    results  = mimic_train_datasets(client,text, top_n=5)

    input_ =    str(j.Symptoms)+ '\n'  + str(j.text) + '\n' + str(disease_codes)
    output = get_icd9_codes_mistral(prompt_creation_rag(input_, results))

    processed_data = preprocess_icd9_codes(output, j.note_id, output_dir='infer_llm_mimic_rag_kg_5')

    if i==500:
        break

client.close()

true_labels = []
pred_labels = []

for i in os.listdir('infer_llm_mimic_rag_kg_5'):
    with open(f'infer_llm_mimic_rag_kg_5/{i}', 'r') as f:
        note_id = i.split('icd9_codes_processed_')[1].split('.json')[0]

        data = json.loads(f.read())
        pred_codes = [i['code'] for i in data['icd9_codes']]


    # print(list(mimic_4_df[mimic_4_df.note_id == note_id].SHORT_CODES)[0])
    true_codes = list(mimic_4_df[mimic_4_df.note_id == note_id].SHORT_CODES)[0].split(',')

    true_labels.append(true_codes)
    pred_labels.append(pred_codes)

len(true_labels), len(pred_labels)

en(true_labels), len(pred_labels)
f1_scores , prec, rec = [], [], []
for true_label,pred_label in zip(true_labels, pred_labels) :
    precision, recall, f1 = calculate_f1(true_label, pred_label)
    f1_scores.append(f1)
    prec.append(precision)
    rec.append(recall)

macro_f1_score = sum(f1_scores) / len(f1_scores)
macro_prec = sum(prec) / len(prec)
macro_rec = sum(rec) / len(rec)
print(f"Macro Precision Score: {macro_prec:.4f}")
print(f"Macro Recall Score: {macro_rec:.4f}")
print(f"Macro F1 Score: {macro_f1_score:.4f}")

