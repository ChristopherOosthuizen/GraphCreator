import markdownify
import urllib.request
import threading
#  Converts content of webpahes into markdown text
def _url_to_md(url):
    html = urllib.request.urlopen(url).read().decode('utf-8')
    return markdownify.markdownify(html, heading_style="ATX")

def _chunkText(text):
    splitter = MarkdownTextSplitter(chunk_size=10000, chunk_overlap=200)
    return splitter.create_documents([text])

def _set_chunk(url, chunk,chunks, position):
    chun = format(chunk,url)
    chunks[position] = chun
def _get_text_chunks(text):
    chunks = _chunkText(text)
    print(len(chunks))
    threads = []
    for x in range(len(chunks)):
        thread = threading.Thread(target=_set_chunk, args=("",chunks[x],chunks,x))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    for x in range(len(chunks)-1,-1,-1):
        if "<#notext#>" in chunks[x]:
            chunks.pop(x)
    return chunks