import csv
import json
import re
import ahocorasick
import logging
import spacy
from negspacy.negation import Negex
from negspacy.termsets import termset
from spacy.language import Language
from collections import defaultdict
from tqdm import tqdm

# Initialize logging
logging.basicConfig(
    filename='umls_negex_processing.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

def load_umls_terms(csv_file):
    """
    Load UMLS terms from a CSV file.
    Each term is converted to lowercase for case-insensitive matching.
    Returns a dictionary mapping terms to a set of CUIs.
    """
    term_to_cuis = defaultdict(set)
    try:
        with open(csv_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cui = row['cui'].strip()
                term = row['term'].strip().lower()
                if term and cui:
                    term_to_cuis[term].add(cui)
        logger.info(f"Loaded {len(term_to_cuis)} unique UMLS terms from '{csv_file}'.")
        return term_to_cuis
    except Exception as e:
        logger.error(f"Error loading UMLS terms from '{csv_file}': {e}")
        raise

def build_automaton(term_to_cuis):
    """
    Build an Aho-Corasick automaton for efficient multi-term searching.
    The automaton maps each term to its associated CUI(s).
    """
    A = ahocorasick.Automaton()
    for term, cuis in term_to_cuis.items():
        A.add_word(term, (term, cuis))
    A.make_automaton()
    logger.info("Aho-Corasick automaton built successfully.")
    return A

def is_whole_word(sentence, start, end):
    """
    Check if the matched term is a whole word in the sentence.
    """
    if start > 0 and re.match(r'\w', sentence[start - 1]):
        return False
    if end < len(sentence) and re.match(r'\w', sentence[end]):
        return False
    return True

def find_terms(sentence, automaton):
    """
    Find all exact whole-word matches of terms in the sentence using the automaton.
    Returns a list of tuples: (term, set_of_cuis, start_index, end_index)
    """
    found = []
    for end_idx, (term, cuis) in automaton.iter(sentence.lower()):
        start_idx = end_idx - len(term) + 1
        if is_whole_word(sentence, start_idx, end_idx + 1):
            found.append((term, cuis, start_idx, end_idx + 1))
    return found

def initialize_spacy():
    """
    Initialize the spaCy model with Negex.
    """
    try:
        
        ts = termset("en")
        if "negex" not in Language.factories:
            @Language.factory("negex")
            def negex_component_function(nlp, name):
                return Negex(nlp, 
                      name = "negex",
                      ent_types=["DIS", "SYM"],
                      neg_termset=ts.get_patterns(),
                      extension_name = "negex",
                      chunk_prefix="B",
                      pseudo_negations = ts["pseudo_negations"],
                      preceeding_negations = ts["preceeding_negations"],
                      following_negations = ts["following_negations"],
                      termination = ts["termination"],
                      )
        nlp = spacy.load("en_core_web_sm")
        if "negex" not in nlp.pipe_names:
            nlp.add_pipe("negex", last=True)
            logger.info("Negex component added to the pipeline.")
        else:
            logger.info("Negex component already exists")


        logger.info("spaCy model with Negex initialized successfully.")
        return nlp
    except Exception as e:
        logger.error(f"Error initializing spaCy model: {e}")
        raise

def process_json(json_file, automaton, nlp, disease_term, output_json, output_csv):
    """
    Process the JSON file, search for terms and disease, apply NegEx,
    and store the results.
    """
    results = []
    symptom_counts = defaultdict(lambda: defaultdict(lambda: {'positive_count': 0, 'negative_count': 0}))
    disease_term_lower = disease_term.lower()

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} JSON objects from '{json_file}'.")
    except Exception as e:
        logger.error(f"Error loading JSON file '{json_file}': {e}")
        raise

    for item in tqdm(data, desc="Processing JSON objects"):
        pmid = item.get('id', '').strip()
        content = item.get('contents', '').strip()
        if not pmid or not content:
            logger.warning(f"Missing 'id' or 'contents' in JSON object: {item}")
            continue

        # Process content with spaCy
        doc = nlp(content)

        for sent in doc.sents:
            sentence = sent.text
            sentence_lower = sentence.lower()

            if disease_term_lower not in sentence_lower:
                continue

            # Find UMLS terms in the sentence
            term_matches = find_terms(sentence, automaton)
            if not term_matches:
                continue
            # Process sentence with spaCy for NegEx
            sent_doc = nlp(sentence)

            # Create a list to store unique terms in the sentence with their CUIs
            unique_terms = defaultdict(set)  # term -> set of cuis
            term_negation = defaultdict(bool)  # term -> is_negated

            for term, cuis, start, end in term_matches:
                unique_terms[term].update(cuis)
                span = sent_doc.char_span(start, end, alignment_mode='expand')
                if span is None:
                    term_negation[term] = False
                    continue
                is_neg = span._.negex
                if is_neg:
                    term_negation[term] = True

            if not unique_terms:
                continue

            # Update symptom counts
            for term, cuis in unique_terms.items():
                for cui in cuis:
                    if term_negation.get(term, False):
                        symptom_counts[cui][term]['negative_count'] += 1
                    else:
                        symptom_counts[cui][term]['positive_count'] += 1

            # Prepare found_terms with cui
            found_terms_with_cui = []
            for term, cuis in unique_terms.items():
                found_terms_with_cui.append({
                    'term': term,
                    'cui': list(cuis)
                })

            # Append the result
            results.append({
                'PMID': pmid,
                'found_sentence': sentence,
                'found_terms': found_terms_with_cui
            })

    logger.info(f"Processed all JSON objects. Total matched sentences: {len(results)}")

    # Write matched sentences to JSON
    try:
        with open(output_json, 'w', encoding='utf-8') as out_f:
            json.dump(results, out_f, indent=2)
        logger.info(f"Matched sentences saved to '{output_json}'.")
    except Exception as e:
        logger.error(f"Error writing to JSON file '{output_json}': {e}")
        raise

    # Write symptom counts to CSV
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['cui', 'symptom', 'positive_count', 'negative_count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for cui, terms in symptom_counts.items():
                for term, counts in terms.items():
                    writer.writerow({
                        'cui': cui,
                        'symptom': term,
                        'positive_count': counts['positive_count'],
                        'negative_count': counts['negative_count']
                    })
        logger.info(f"Symptom counts saved to '{output_csv}'.")
    except Exception as e:
        logger.error(f"Error writing to CSV file '{output_csv}': {e}")
        raise

def main():
    # File paths
    umls_csv_file = 'umls_terms_T184.csv'
    output_folder = '/mnt/0C6C8FC06C8FA2D6/sparse_results_3digit'

    logger.info("Script started.")

    # Load UMLS terms
    logger.info("Loading UMLS terms...")
    term_to_cuis = load_umls_terms(umls_csv_file)

    # Build Aho-Corasick automaton
    logger.info("Building Aho-Corasick automaton...")
    automaton = build_automaton(term_to_cuis)

    # Initialize spaCy with NegEx
    logger.info("Initializing spaCy model with NegEx...")
    nlp = initialize_spacy()

    import os
    folder_path = '/mnt/0C6C8FC06C8FA2D6/sparse_retrieval_3digit_ICD'

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            json_file = os.path.join(folder_path, file_name)


            logger.info(f"Processing {json_file} JSON file and searching for terms...")
            disease_term = file_name.replace('_', ' ').replace('.json', '')
            output_json = os.path.join(output_folder, f"{file_name.replace('.json', '')}_sentences.json")
            output_csv = os.path.join(output_folder, f"{file_name.replace('.json', '')}_symptom_counts.csv")
            process_json(json_file, automaton, nlp, disease_term, output_json, output_csv)

            logger.info(f"{json_file} Processing completed successfully.")
    logger.info("Script completed.")


if __name__ == "__main__":
    main()
