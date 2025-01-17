import requests
import os
from downloader_service.gcp_handler import GCPStorageManager
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class PaperDownloader:
    def __init__(self, api_key, credentials_path, bucket_name):
        self.api_key = api_key
        self.bucket_name = bucket_name
        self.credentials_path = credentials_path
        self.headers = {'x-api-key': self.api_key}
        
        # Initialize GCPStorageManager
        self.gcp_manager = GCPStorageManager(credentials_path)
        
        # Create a session with retry strategy
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

    def request_paper_details(self, query, offset):
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        query_params = {
            'query': query,
            'limit': 100,
            'offset': offset,
            'fields': 'isOpenAccess,openAccessPdf,abstract,url,title,citationCount,year,fieldsOfStudy,s2FieldsOfStudy',
        }
        try:
            response = self.session.get(url, params=query_params, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.json() if response.status_code == 200 else None
        except requests.exceptions.RequestException as e:
            print(f"Request failed with an exception: {e}")
            return None

    def get_paper_details(self, query, n):
        offset = 0
        papers = []
        while len(papers) < n:
            response = self.request_paper_details(query, offset)
            if response and response.get('data'):
                papers.extend(response['data'])
                offset += 100
            else:
                break
        return papers

    def filter_open_access_papers(self, papers):
        open_access_papers = [paper for paper in papers if paper.get('openAccessPdf')]
        open_access_papers.sort(key=lambda x: x['citationCount'], reverse=True)
        return open_access_papers[:20]

    def upload_pdf_content_to_gcs(self, pdf_content, gcs_file_path):
        return self.gcp_manager.upload_content(self.bucket_name, pdf_content, gcs_file_path)

    def download_top_papers(self, query, n=1000, limit=20, directory=""):
        papers = self.get_paper_details(query, n)
        top_open_access_papers = self.filter_open_access_papers(papers)
        downloaded_papers = []
        downloaded_count = 0
        for idx, paper in enumerate(top_open_access_papers):
            if downloaded_count >= limit:
                break
            pdf_url = paper['openAccessPdf']['url']
            gcs_save_path = f"{directory}/{paper['title']}.pdf"
            try:
                response = self.session.get(pdf_url)
                response.raise_for_status()
                uploaded_url = self.upload_pdf_content_to_gcs(response.content, gcs_save_path)
                downloaded_count += 1
                downloaded_papers.append({
                    "title": paper['title'],
                    "file_path": uploaded_url
                })
                print(f"Downloaded {paper['title']} and uploaded to {uploaded_url}")
            except requests.exceptions.RequestException as e:
                print(f"Failed to download PDF from {pdf_url}. Error: {e}")

        return downloaded_papers
