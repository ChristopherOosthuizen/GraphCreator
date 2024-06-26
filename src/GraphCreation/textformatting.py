import markdownify
import urllib.request
import threading
from  langchain.text_splitter import MarkdownTextSplitter 
import os 
from pdfminer.high_level import extract_text
from . import LLMFunctions as LLM
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
prompts_dir = os.path.join(current_dir, 'prompts')
import tiktoken
from bs4 import BeautifulSoup
import pandas as pd
os.environ['TOKENIZERS_PARALLELISM'] = "true"
def format_text(prompt, url, pipeline_id=0):
    return LLM.generate_chat_response( open( os.path.join(prompts_dir,"formatting")).read(),prompt, pipeline_id)

from llmlingua import PromptCompressor
import torch 
device_map = ""
if torch.cuda.is_available():
    torch.set_default_device("cuda")
    device_map = "cuda"
elif torch.backends.mps.is_available():
    torch.set_default_device("mps")
    device_map = "mps"
else:
    torch.set_default_device("cpu")
    device_map = "cpu"
llm_lingua = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True,
    device_map=device_map,
)
def extract_relevant_text(html: str) -> str:
    # Parse the HTML
    soup = BeautifulSoup(html, 'html.parser')

    # Extract the relevant text
    relevant_text = []

    # Check for the specific text of interest
    site_sub = soup.find(id='siteSub')

    if site_sub:
        relevant_text.append(site_sub.get_text(strip=False))

    paragraphs = soup.find_all(['p',"h1",'h2','h3','h4','h5','h6','ol','dl'])
    for p in paragraphs:
        relevant_text.append(p.get_text(strip=False))  # Add a space after each extracted text segment

    # Extract alt descriptions of images
    images = soup.find_all('img')
    for img in images:
        alt = img.get('alt')
        if alt:
            relevant_text.append(alt)
    splitter = MarkdownTextSplitter(chunk_size=512, chunk_overlap=0)
    result_text = " ".join(relevant_text)
    splits = splitter.create_documents([result_text])
    for x in range(len(splits)):
        splits[x] = str(splits[x])
    if len(splits) == 0:
        return ""
    
    if len(splits) == 1:
        return splits[0]
    compressed_prompt = llm_lingua.compress_prompt(
        context=list(splits),
        rate=0.33,
        force_tokens=["!", ".", "?", "\n"],
        drop_consecutive=True,
    )
    prompt = "\n\n".join([compressed_prompt["compressed_prompt"]])
    return prompt
def get_tables_from_url(url):
    html = urllib.request.urlopen(url).read().decode('utf-8')
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all("table")

    # List to hold dataframes
    dataframes = []

    # Loop through all tables found and convert to DataFrame
    for index, table in enumerate(tables):
        try:
            df = pd.read_html(str(table))[0]
            dataframes.append(df)
        except:
            print(str(table))

    return dataframes

def get_triplets_from_table(df):
    triplets = []
    for i, row in df.iterrows():
        username= row[0]
        if str(username) == "nan":
            continue
        for col in df.columns:
            if str(row[col]) != "nan":
                triplets.append(f"{username},{col},{row[col]}")
                triplets.append(f"{username},positionInList,{i}")
    
    return "\n".join(triplets).lower()
            
def url_to_md(url):
    html = urllib.request.urlopen(url).read().decode('utf-8')
    html= extract_relevant_text(html)
    return html
def pdf_to_md(file):
    relevant_text = extract_text(file)
    splitter = MarkdownTextSplitter(chunk_size=510, chunk_overlap=0)
    result_text = " ".join(relevant_text)
    splits = splitter.create_documents([result_text])
    for x in range(len(splits)):
        splits[x] = str(splits[x])
    if len(splits) == 0:
        return ""
    
    if len(splits) == 1:
        return splits[0]
    compressed_prompt = llm_lingua.compress_prompt(
        context=list(splits),
        rate=0.33,
        force_tokens=["!", ".", "?", "\n"],
        drop_consecutive=True,
    )
    prompt = "\n\n".join([compressed_prompt["compressed_prompt"]])
    return prompt

def folder_to_md(folder):
    files = os.listdir(folder)
    result = []
    for file in files:
        if file.endswith(".pdf"):
            result.append(pdf_to_md(folder+"/"+file))
    return "\n\n".join(result)
def chunk_text(text):
    splitter = MarkdownTextSplitter(chunk_size=200, chunk_overlap=25)
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

enc = tiktoken.encoding_for_model("gpt-4o")
def token_compression(text_list):
    tokens = []
    words = []
    taken = {}
    for x in range(len(text_list)):
        text_list[x] = text_list[x].replace("-","").replace("\"","").replace("\'","").strip()
        encoding = enc.encode(text_list[x].replace(" ",""))
        if len(encoding) == 1:
            taken[encoding[0]] = text_list[x]
        else:
            tokens.append(encoding)
            words.append(text_list[x])
    
    for x in range(len(tokens)):
        for y in range(len(tokens[x])):
            if not tokens[x][y] in taken:
                taken[tokens[x][y]] = words[x]
                break
    reformatted = {}
    for x in taken:
        reformatted[enc.decode([x])] = taken[x].replace("\\","").strip()
    return reformatted

def decompress(text, token_dict):
    result = []
    for x in text.split("\n"):
        objects = x.replace("-","").replace("\"","").replace("\'","").strip().split(",")
        if len(objects) == 3:
            objects[0] = objects[0].strip()
            objects[1] = objects[1].strip()
            objects[2] = objects[2].strip()
            obj1 = ""
            if objects[0] in token_dict:
                obj1 = token_dict[objects[0]]
            else:
                obj1 = objects[0]
            obj0 = ""
            if objects[1] in token_dict:
                obj0 = token_dict[objects[1]]
            else:
                obj0 = objects[1]
            obj2 = ""
            if objects[2] in token_dict:
                obj2 = token_dict[objects[2]]
            else:
                obj2 = objects[2]
            result.append(f"{obj1},{obj0},{obj2}")
    return "\n".join(result)
def expand_compress(new_list, token_dict):
    new_dict = {}
    for x in token_dict.keys():
        new_dict[enc.encode(x)[0]] = token_dict[x]

    values = list(new_dict.values()) 
    for x in range(len(new_list)):
        if not new_list[x] in values and not new_list[x] in token_dict:
            tokens = enc.encode(new_list[x].replace(" ","").replace("\"","").replace("\"","").strip())
            for y in range(len(tokens)):
                if not tokens[y] in new_dict:
                    new_dict[tokens[y]] = new_list[x]
                    break
    reformatted = {}
    for x in new_dict.keys():
        reformatted[enc.decode([x])] = new_dict[x].replace("\\","").strip()

    return reformatted

def unroll_triplets(text):
    result = []
    for x in text.split("\n"):
        objects = x.split(",")
        if len(objects) == 3:
            result.append(objects[0])
            result.append(objects[1])
            result.append(objects[2])

    return "\n".join(result)
def compress(text, token_dict):
    result = []
    for x in text.split("\n"):
        objects = x.split(",")
        if len(objects) == 3:
            obj1 = ""
            if objects[0].strip() in token_dict.values():
                obj1 = list(token_dict.keys())[list(token_dict.values()).index(objects[0])]
            else:
                obj1 = objects[0]
            obj0 = ""
            if objects[1].strip() in token_dict.values():
                obj0 = list(token_dict.keys())[list(token_dict.values()).index(objects[1])]
            else:
                obj0 = objects[1]
            obj2 = ""
            if objects[2].strip() in token_dict.values():
                obj2 = list(token_dict.keys())[list(token_dict.values()).index(objects[2])]
            else:
                obj2 = objects[2]
            result.append(f"{obj1},{obj0},{obj2}")
    return "\n".join(result)