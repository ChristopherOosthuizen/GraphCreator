from langchain.text_splitter import MarkdownTextSplitter
from thefuzz import fuzz
from openai import OpenAI
import json
import threading
import networkx as nx

def _format(prompt,url):
    client = OpenAI()
    response = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {"role": "system", "content": "You are a text filtration system, you are given a short blurb of text and its your job to determine weather this is irrelevant information from a text page for formatting or system headers or if its the main content of the webpage. given the url name. and content."},
    {"role": "user", "content": f"""url: {url}
content: {prompt}
After reviewing the content return nothing but the text unfiltered if there is no relevant content simply return no text. Reformat your output so its easily readable and remove formatting problems.If there is no content relevant return this string of text <#notext#>."""},
    ]
    )
    return response.choices[0].message.content

def _generate(system_promt,user_prompt):
    client = OpenAI()
    response = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {"role": "system", "content": system_promt},
    {"role": "user", "content": user_prompt},
    ]
    )
    return response.choices[0].message.content

def _create_knowledge_triplets(text_chunk="", repeats=5):
    system_prompt = open("prompts/TripletCreationSystem").read()

    prompt =f"Context: ```{text_chunk}``` \n\nOutput: "
    response = str(_generate(system_prompt,prompt))
    for x in range(repeats):
        system_prompt = open("prompts/TripletCreationSystem").read()
        prompt="""Here is the prompt updated to insert additional triplets into the existing ontology:
Read this context carefully and extract the key concepts and relationships discussed:"""+text_chunk+"""Here is the ontology graph generated from the above context:"""+response+open("prompts/TripletIterationStandard").read()
        response = str(_generate(system_prompt,prompt))
    response = str(_fix_ontology(response))
    response = response[response.find("["):response.find("]")+1]
    response = response.replace("node1","node_1")
    response = response.replace("node2","node_2")
    return response

def _fix_format(input):
    prompt = f"given this json \nOriginal Json: {input}"+open("prompts/TripletCreationSystem").read()
    response = str(_generate("",prompt))
    response = response[response.find("["):response.find("]")+1]
    return response

import pandas as pd
import networkx as nx
from  itertools import combinations
def _triplets_to_json(triplets):
    df = pd.DataFrame({"node_1":[],"node_2":[],"edge":[]})
    for triplet in triplets:
        df = df._append({"node_1":triplet[0],"node_2":triplet[2],"edge":triplet[1]},ignore_index=True)
    return json.dumps(df.to_dict(orient="records")).replace(",",",\n")
def _ontologies_to_unconncected(ont1, ont2):
    if ont1.strip() == "":
        return ont2
    if ont2.strip() == "":
        return ont1
    try:
        ont1 = json.loads(ont1)
    except:
        ont1 = json.loads(_fix_format(ont1))
    try:
        ont2 = json.loads(ont2)
    except:
        ont2 = json.loads(_fix_format(ont2))
    for triplet in ont2:
        if triplet not in ont1:
            ont1.append(triplet)
    df = pd.DataFrame(ont1)
    G = nx.Graph()
    for x in df.iloc:
        G.add_edge(x["node_1"],x["node_2"],label=x["edge"])
    dis = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    result = []
    disconnected = []
    for x in dis:
        if x.number_of_nodes() <= 1:
            continue
        res = []
        for u,v,data in x.edges(data=True):
            res.append((u,data["label"],v))
        disconnected.append(res)
    for x in disconnected:
        result.append(_triplets_to_json(x))
    return result

def _combine_ontologies(ont1, ont2, sums):
    disconnected = "\n\n".join(_ontologies_to_unconncected(ont1, ont2))
    prompt = f"""Here's a prompt that takes a series of unconnected ontology graphs and their corresponding context chunks, and generates a single unified ontology in JSON format that combines the individual ontologies with additional linking triplets:
Given a series of unconnected ontology graphs extracted from their respective context chunks, your task is to analyze the contexts and identify potential relationships that could link these isolated ontologies to form a single, cohesive knowledge graph. The goal is to create a unified ontology where all the individual ontologies are connected through meaningful semantic relationships.
Please provide the context chunks and their corresponding ontologies in the following format:
Context chunk:"""+sums+"""Ontologys:"""+disconnected+"""Please follow these steps to create the unified ontology:"""+open("prompts/CombineOntologies").read()

    response = str(_generate("",prompt))
    response = response[response.find("["):response.find("]")+1].lower()
    return response

def _one_switch(ont):
    try:
        ont = json.loads(ont)
    except:
        ont = json.loads(_fix_format(ont))
    df = pd.DataFrame(ont)
    G = nx.Graph()
    for x in df.iloc:
        G.add_edge(x["node_1"],x["node_2"],label=x["edge"])
    dis = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    result = []
    disconnected = []
    for x in dis:
        if x.number_of_nodes() <= 1:
            continue
        res = []
        for u,v,data in x.edges(data=True):
            res.append((u,data["label"],v))
        disconnected.append(res)
    for x in disconnected:
        result.append(_triplets_to_json(x))
    return result

def _fix_ontology(ont, context):
    disconnected = "\n\n".join(_one_switch(ont))
    prompt = f"""Here's a prompt that takes a series of unconnected ontology graphs and their corresponding context chunks, and generates a single unified ontology in JSON format that combines the individual ontologies with additional linking triplets:
Given a series of unconnected ontology graphs extracted from their respective context chunks, your task is to analyze the contexts and identify potential relationships that could link these isolated ontologies to form a single, cohesive knowledge graph. The goal is to create a unified ontology where all the individual ontologies are connected through meaningful semantic relationships.
Please provide the context chunks and their corresponding ontologies in the following format:
Context chunk:"""+context+"""Ontologys:"""+disconnected+"""Please follow these steps to create the unified ontology:"""+open("prompts/fixOntologies").read()
    response = str(_generate("",prompt))
    response = response[response.find("["):response.find("]")+1].lower()
    return response


def new_summary_prompt(summary, text_chunk):
    summary_prompt = open("prompts/summaryPrompt").read()
    summary = _generate(summary_prompt, f"existing_summary: {summary} new_text_chunk: {text_chunk}")
    return summary
def _make_one_triplet(list, position, chunk):
    list[position] = _create_knowledge_triplets(text_chunk=chunk)
def _combine_one(ont1, ont2, sum1,sum2, list, position, summaries):
    sums = new_summary_prompt(sum1,sum2)
    summaries[position] = sums
    list[position] = _combine_ontologies(ont1, ont2, sums)

def _create_kg(chunks, repeats=5, converege=True):
    
    print(f"Number of chunks: {len(chunks)}")
    triplets = []
    combinations = [] 
    if len(chunks) == 1:
        return [_create_knowledge_triplets(text_chunk=chunks[0])]
    combinations = []
    summaries = []
    threads = []
    triplets = [""]*len(chunks)
    for x in range(len(chunks)):
        thread = threading.Thread(target=_make_one_triplet, args=(triplets,x,chunks[x]))
        threads.append(thread)
        thread.start()
    for thread in threads:
        print(x/len(chunks))
        thread.join()
    combinations = triplets
    summaries = chunks    
    for x in range(repeats):
        if len(combinations) == 1:
            break
        old_combinations = combinations
        old_summaries = summaries
        summaries = [""]*(int(len(combinations)/2)+len(combinations)%2)
        combinations = [""]*(int(len(combinations)/2)+len(combinations)%2)
        threads = []
        for x in range(1,len(old_combinations),2):
            thread = threading.Thread(target=_combine_one, args=(old_combinations[x-1],old_combinations[x],old_summaries[x-1],old_summaries[x],combinations,x//2,summaries))
            threads.append(thread)
            thread.start()
        for thread in threads:
            print(1/len(combinations))
            thread.join()
        if len(combinations)%2 == 1:
            combinations[-1] = old_combinations[-1]
            summaries[-1] = old_summaries[-1]
    if converege:
        while len(_ontologies_to_unconncected(combinations[0], combinations[0])) > 1:
            print(len(_ontologies_to_unconncected(combinations[0], combinations[0])))
            combinations[0] = _fix_ontology(combinations[0], summaries[0])
        return combinations

    return combinations

from llama_index.core import set_global_tokenizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from llama_index.core import StorageContext
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import Settings
from llama_index.core.chat_engine import ContextChatEngine
def graphquestions(graph, prompt):
    graph_store = SimpleGraphStore()
    for x in graph.edges(data=True):
        graph_store.upsert_triplet(x[0],x[2]['title'],x[1])
        storage_context = StorageContext.from_defaults(graph_store=graph_store)
    graph_rag_retriever = KnowledgeGraphRAGRetriever(
        storage_context=storage_context,
        verbose=False,
    )
    chat_engine = ContextChatEngine.from_defaults(
        retriever=graph_rag_retriever,
        verbose=False,
        max_entities= 30,
        graph_traversal_depth=10,
        max_knowledge_sequence=15,
    )
    response = chat_engine.chat(
        prompt,
    )
    return response.response
import textformatting
def create_KG_from_text(text, output_file="./output/"):
    jsons = _create_kg(textformatting._get_text_chunks(text), converege=False, repeats=1)
    Graph = nx.Graph()
    for x in jsons:
        try:
            x = json.loads(x)
        except:
            print(jsons)
            x = json.loads(_fix_format(x))
        for y in x:
            Graph.add_edge(y["node_1"],y["node_2"],label=y["edge"])

    nx.write_graphml(Graph, output_file+"graph.graphml")
    open(output_file+"graph.json","w").write(json.dumps(jsons))
    from pyvis.network import Network
    nx.draw(Graph, with_labels = True)
    nt = Network('100%', '100%')
    nt.from_nx(Graph)
    nt.show(output_file+"graph.html", notebook=False)  
    return Graph

def create_KG_from_url(url, output_file="./output/"):
    text = textformatting._url_to_md(url)
    jsons = create_KG_from_text(text, output_file)
    return jsons
from pdfminer.high_level import extract_text

def _convert_to_markdown(text):
    lines = text.split("\\\\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.isupper() and len(stripped) < 50:
            lines[i] = f"## {stripped}"
    return "\\\\n".join(lines)

def create_KG_from_pdf(pdf, output_file="./output/"):
    text = _convert_to_markdown(extract_text(pdf))
    print(text)
    jsons = create_KG_from_text(text, output_file)
    return jsons
