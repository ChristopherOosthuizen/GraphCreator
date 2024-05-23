import markdownify
import urllib.request
import threading
from  langchain.text_splitter import MarkdownTextSplitter 
from openai import OpenAI

def format_text(prompt, url):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a text filtration system, you are given a short blurb of text and its your job to determine weather this is irrelevant information from a text page for formatting or system headers or if its the main content of the webpage. given the url name. and content."},
            {"role": "user", "content": f"url: {url} Prompt: {prompt}"}
        ]
    )
    return response.choices[0].message.content

def url_to_md(url):
    html = urllib.request.urlopen(url).read().decode('utf-8')
    return markdownify.markdownify(html, heading_style="ATX")

def chunk_text(text):
    splitter = MarkdownTextSplitter(chunk_size=10000, chunk_overlap=200)
    return splitter.create_documents([text])

def set_chunk(url, chunk, chunks, position):
    chun = format_text(chunk, url)
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

def _convert_to_markdown(text):
    lines = text.split("\\\\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.isupper() and len(stripped) < 50:
            lines[i] = f"## {stripped}"
    return "\\\\n".join(lines)