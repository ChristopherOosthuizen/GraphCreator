import markdownify
import urllib.request
import threading
from  langchain.text_splitter import MarkdownTextSplitter 
import os 
from pdfminer.high_level import extract_text
from . import LLMFunctions as LLM
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
prompts_dir = os.path.join(current_dir, '..', 'prompts')
def format_text(prompt, url, pipeline_id=0):
    return LLM.generate_chat_response( open( os.path.join(prompts_dir,"formatting")).read(),prompt, pipeline_id)

def extract_relevant_text(html: str) -> str:
    from bs4 import BeautifulSoup

    # Parse the HTML
    soup = BeautifulSoup(html, 'html.parser')

    # Extract the relevant text
    relevant_text = []

    # Check for the specific text of interest
    site_sub = soup.find(id='siteSub')
    content_text = soup.find(id='mw-content-text')

    if site_sub:
        relevant_text.append(site_sub.get_text(strip=False))

    if content_text:
        paragraphs = content_text.find_all('p')
        for p in paragraphs:
            relevant_text.append(p.get_text(strip=False))  # Add a space after each extracted text segment

    result_text = " ".join(relevant_text)
    return result_text

def url_to_md(url):
    html = urllib.request.urlopen(url).read().decode('utf-8')
    html= extract_relevant_text(html)
    return html
def pdf_to_md(file):
    return extract_text(file)
def chunk_text(text):
    text = text.replace("\n\n", "\n")
    text = text.replace("[", " ")
    text = text.replace("]", " ")
    text = text.replace("{", " ")
    text = text.replace("}", " ")
    splitter = MarkdownTextSplitter(chunk_size=600, chunk_overlap=200)
    splits = splitter.create_documents([text])
    for x in range(len(splits)):
        splits[x] = str(splits[x])
    return splits

def set_chunk(url, chunk, chunks, position):
    if 'KG_GPUS' not in os.environ:
        os.environ['KG_GPUS'] = '0'
    gpu_length = len(os.environ['KG_GPUS'].split(","))
    chun = format_text(chunk, url, pipeline_id=position%gpu_length)
    chunks[position] = chun

def get_text_chunks(text):
    chunks = chunk_text(text)
    threads = []
    for x in range(len(chunks)):
        thread = threading.Thread(target=set_chunk, args=("", chunks[x], chunks, x))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    for x in range(len(chunks)-1, -1, -1):
        if "<#notext#>" in chunks[x]:
            chunks.pop(x)
    return chunks
