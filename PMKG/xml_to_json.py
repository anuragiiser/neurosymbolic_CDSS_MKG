import os
import gzip
import xml.etree.ElementTree as ET
import csv
import json

# Define paths
input_folder = '/mnt/0C6C8FC06C8FA2D6/medline_data'
output_folder_csv = '/mnt/0C6C8FC06C8FA2D6/output/csv'
output_folder_json = '/mnt/0C6C8FC06C8FA2D6/output/json'

# Ensure output folder exists
os.makedirs(output_folder_csv, exist_ok=True)
os.makedirs(output_folder_json, exist_ok=True)

# Function to extract data from a single XML file
def extract_data_from_xml(gzip_file):
    data = []
    try:
        with gzip.open(gzip_file, 'rb') as f:
            tree = ET.parse(f)
            root = tree.getroot()

            for article in root.findall('.//PubmedArticle'):
                medline = article.find('MedlineCitation')
                if medline is None:
                    continue

                pmid_elem = medline.find('PMID')
                pmid = pmid_elem.text.strip() if pmid_elem is not None else ''

                article_elem = medline.find('Article')
                if article_elem is not None:
                    title_elem = article_elem.find('ArticleTitle')
                    title = title_elem.text.strip() if title_elem is not None else ''

                    abstract_elem = article_elem.find('Abstract')
                    if abstract_elem is not None:
                        # Some abstracts have multiple AbstractText elements
                        abstract_texts = [ab.text.strip() for ab in abstract_elem.findall('AbstractText') if ab.text]
                        abstract = ' '.join(abstract_texts)
                    else:
                        abstract = ''
                else:
                    title = ''
                    abstract = ''

                data.append({
                    'PMID': pmid,
                    'Title': title,
                    'Abstract': abstract
                })
    except Exception as e:
        print(f"Error processing {gzip_file}: {e}")
    return data

# Function to save data to CSV
def save_to_csv(data, csv_file):
    try:
        with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=['PMID', 'Title', 'Abstract'])
            writer.writeheader()
            for entry in data:
                writer.writerow(entry)
    except Exception as e:
        print(f"Error writing to CSV {csv_file}: {e}")

# Function to save data to JSON
def save_to_json(data, json_file):
    try:
        with open(json_file, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error writing to JSON {json_file}: {e}")

# Process each file directly without verifying integrity
for i in range(1, 1220):  # Adjust the range according to the number of files
    file_number = str(i).zfill(4)
    gzip_file = os.path.join(input_folder, f'pubmed24n{file_number}.xml.gz')

    if os.path.exists(gzip_file):
        print(f'Processing {gzip_file}...')

        # Extract data from the current file
        file_data = extract_data_from_xml(gzip_file)

        if file_data:
            # Define output file names based on the current file being processed
            output_csv = os.path.join(output_folder_csv, f'pubmed24n{file_number}.csv')
            output_json = os.path.join(output_folder_json, f'pubmed24n{file_number}.json')

            # Save the extracted data to separate CSV and JSON files
            save_to_csv(file_data, output_csv)
            save_to_json(file_data, output_json)

            print(f'Saved: {output_csv} and {output_json}')
        else:
            print(f'No data extracted from {gzip_file}.')
    else:
        print(f'{gzip_file} does not exist. Skipping...')
