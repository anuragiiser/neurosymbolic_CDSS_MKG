import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# URL of the page containing the links
url = 'https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/'

# Destination folder to save the downloaded files
destination_folder = 'medline_data'

# Create the folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Send a GET request to the URL
response = requests.get(url)
response.raise_for_status()  # Raise an error for bad status codes

# Parse the HTML content of the page
soup = BeautifulSoup(response.content, 'html.parser')

# Find all <a> tags
links = soup.find_all('a')

# Download each XML link
for link in links:
    href = link.get('href')
    if href and (href.endswith('.xml') or 'xml' in href):
        full_url = urljoin(url, href)  # Handle relative URLs
        file_name = os.path.join(destination_folder, os.path.basename(full_url))
        print(f'Downloading {full_url} to {file_name}')
        try:
            file_response = requests.get(full_url)
            file_response.raise_for_status()
            with open(file_name, 'wb') as file:
                file.write(file_response.content)
            print(f'Successfully downloaded {file_name}')
        except requests.exceptions.RequestException as e:
            print(f'Failed to download {full_url}: {e}')
