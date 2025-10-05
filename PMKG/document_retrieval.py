import json
import os
import numpy as np
import pandas as pd
from pyserini.search.lucene import LuceneSearcher

# Initialize searcher with the indexed data
index_dir = '/mnt/0C6C8FC06C8FA2D6/indexes/pubmed-index'
searcher = LuceneSearcher(index_dir)

# Set BM25 parameters (if needed)
searcher.set_bm25(k1=0.9, b=0.4)
bm25_threshold = 9.8

# Function to calculate the dynamic BM25 threshold (sweet spot) based on CFD 
def calculate_dynamic_threshold(scores): 
    if len(scores) < 5:
        bm25_threshold = 9.8
    sorted_scores = np.sort(scores) 
    cumulative_frequencies = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores) 
    # Find the "sweet spot" based on the CFD curve (inflection point) 
    # You can define your own criteria for the sweet spot, e.g., the score at 90% cumulative frequency 
    inflection_point = np.argmax(np.gradient(cumulative_frequencies)) 
    
    # Simple gradient-based approach 
    bm25_threshold = sorted_scores[inflection_point] 
    # Use the score at the inflection point 
    return bm25_threshold


def search_disease(disease_name, k=10):
    hits = searcher.search(disease_name, k=k)
    bm25_scores = [hit.score for hit in hits]
    bm25_threshold = calculate_dynamic_threshold(bm25_scores)
    print(bm25_threshold)
    results = []
    for i in range(len(hits)):
        if hits[i].score>=bm25_threshold:
            doc = searcher.doc(hits[i].docid).raw()
            results.append(json.loads(doc))
    return results

# Search for articles related to a specific disease
dis_file="three_digit_icd9_codes.csv"
dis_df=pd.read_csv(dis_file, delimiter='|', on_bad_lines='skip')
dis_term=dis_df["Description"]
# dis_term=dis_term.loc[12120:]
print(dis_term.head())

for disease in dis_term:
    # disease = 'Carotid sinus syndrome'
    print(disease)
    top_k = 3000
    results = search_disease(disease, k=top_k)
    
    if(len(results))==0:
        continue

    term_name = "_".join(disease.split(" "))
    with open(os.path.join("/mnt/0C6C8FC06C8FA2D6/sparse_retrieval_3digit_ICD", f'{term_name}.json'), 'w') as fp:
        json.dump(results, fp, indent=4)


# # Display top results
# for result in results:
#     print(f"PMID: {result['id']}")
#     # print(f"Title: {result['contents'].split('.')[0]}")
#     print(f"Title&Abstract: {result['contents']}")
#     print()
