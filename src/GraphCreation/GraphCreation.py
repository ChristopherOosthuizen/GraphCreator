import markdownify
import urllib.request
from langchain.text_splitter import MarkdownTextSplitter
import tiktoken
import os
from thefuzz import fuzz

from openai import OpenAI
import re
import json
import threading




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

def _create_knowledge_triplets(text_chunk="", repeats=5):
    system_prompt = """You are an AI assistant trained in extracting ontologies and relationships between concepts from a given context using techniques from category theory and knowledge representation. Your task is to analyze a provided text passage (delimited by ```) and identify the key terms, entities, and concepts discussed, focusing on well-defined and widely used terminology related to materials, systems, processes and methods.
For each pair of identified terms/concepts (up to around 10 pairs total), output a JSON object specifying the two terms as "node_1" and "node_2", and concisely describe the relationship or connection between them in the "edge" field. The goal is to produce a list of JSON objects representing a graph-like ontology, where the nodes are key concepts and the edges define their relationships as stated or implied by the given context.
The output format should be a JSON list, with each element being an object like:
[
{
"node_1": "First concept from extracted ontology",
"node_2": "Second concept that relates to node_1",
"edge": "Concise description of relationship between node_1 and node_2"
},
{
"node_1": "Another key concept",
"node_2": "A different concept that node_1 connects to",
"edge": "Stated or implied relationship between these concepts"
},
{...}
]
Here are two examples of the input context and expected output format:
Context:
```Alice is Marc's mother.```
Output:
[
{
"node_1": "Alice",
"node_2": "Marc",
"edge": "is mother of"
}
]
Context:
```Silk is a strong natural fiber used to catch prey in a web. Beta-sheets control its strength.```
Output:
[
{
"node_1": "silk",
"node_2": "fiber",
"edge": "is a type of"
},
{
"node_1": "silk",
"node_2": "web",
"edge": "is used to make"
},
{
"node_1": "web",
"node_2": "prey",
"edge": "catches"
},
{
"node_1": "beta-sheets",
"node_2": "silk",
"edge": "provide strength to"
}
]
Carefully analyze the provided context and extract around 10 key concepts and relationships between them to produce an ontology graph, ensuring the outputs are consistent and relevant to the topic discussed."""

    prompt =f"Context: ```{text_chunk}``` \n\nOutput: "
    response = str(_generate(system_prompt,prompt))
    for x in range(repeats):
        system_prompt = """You are an AI assistant trained in extracting ontologies and relationships between concepts from a given context using techniques from category theory and knowledge representation. Your task is to analyze a provided text passage (delimited by ```) and identify the key terms, entities, and concepts discussed, focusing on well-defined and widely used terminology related to materials, systems, processes and methods.
For each pair of identified terms/concepts (up to around 10 pairs total), output a JSON object specifying the two terms as "node_1" and "node_2", and concisely describe the relationship or connection between them in the "edge" field. The goal is to produce a list of JSON objects representing a graph-like ontology, where the nodes are key concepts and the edges define their relationships as stated or implied by the given context.
The output format should be a JSON list, with each element being an object like:"""
        prompt="""Here is the prompt updated to insert additional triplets into the existing ontology:
Read this context carefully and extract the key concepts and relationships discussed:"""+text_chunk+"""Here is the ontology graph generated from the above context:"""+response+"""Please augment this ontology graph by adding relevant new concept and relationship triplets while keeping the original ones intact:

Identify important concepts or relationships that were not captured in the original ontology and add them as new node-edge-node triplets.
Ensure any new nodes use standard, unambiguous terminology specific to the materials science domain, avoiding overly generic or vague labels.
For new edges, concisely describe the semantic relationships between concepts as conveyed by the context. Aim for clarity and specificity.
Add the new triplets to the existing JSON list, maintaining alphabetical order of the "node_1" field for readability.
Preserve all of the original ontology triplets in the output, even if some nodes or edges seem redundant with the newly added ones.
The goal is to produce an augmented ontology graph that captures as much of the key knowledge from the context as possible, while avoiding tangential or irrelevant concepts.

Please output the expanded ontology using the same JSON list format as the original, with each dictionary representing a semantic link between two concepts:
[
{
"node_1": "First concept from original ontology",
"node_2": "Related original concept",
"edge": "Original relationship label"
},
{
"node_1": "New concept to add to ontology",
"node_2": "Another new concept or existing one that relates to node_1",
"edge": "Semantic relationship between new node_1 and node_2"
},
{
"node_1": "Another original concept",
"node_2": "Related original concept",
"edge": "Original relationship label"
},
{
"node_1": "Additional new concept to include",
"node_2": "Related new or existing concept",
"edge": "Specific relationship based on context"
},
{...}
]
The output should contain all of the original ontology triplets interleaved with the newly added ones, organized alphabetically by the "node_1" field. This will provide a more comprehensive knowledge graph capturing the main concepts and relationships discussed in the text."""
        response = str(_generate(system_prompt,prompt))
    prompt = """Here is the prompt updated to insert additional triplets into the existing ontology:
Read this context carefully and extract the key concepts and relationships discussed:"""+text_chunk+"""Here is the ontology graph generated from the above context:"""+response+"""Here's a prompt that expands on the previous one by adding triplets to connect all nodes in the ontology graph:
Read this context carefully and extract the key concepts and relationships discussed:
{input}
Here is the ontology graph generated from the above context:
{response}
Please enhance this ontology graph by inserting additional concept and relationship triplets to bridge any gaps and ensure all nodes are connected:

Identify nodes that are currently isolated or disconnected from the main graph structure.
Analyze the context to find semantic relationships that could link these isolated nodes to other concepts in the graph. Focus on the most direct and relevant connections.
Add new triplets to the JSON list to represent these bridging relationships, ensuring the "node_1" and "node_2" fields reference existing nodes in the ontology.
For any new edges added, provide concise, unambiguous labels that accurately capture the nature of the relationship between the connected concepts, as implied by the original context.
If necessary, introduce additional intermediate nodes to link distant concepts together. Ensure these new nodes are labeled with standard, domain-specific terminology.
Aim to create the minimal number of new triplets needed to form a fully connected graph, where there is a path between any two nodes.
Preserve all existing ontology triplets in the output, maintaining alphabetical order by the "node_1" field.

The goal is to produce an augmented ontology graph that not only captures the key concepts and relationships from the context, but also connects them into a cohesive knowledge structure. The resulting graph should allow a domain expert to traverse between any two concepts by following the semantic links.
Please output the expanded ontology using the same JSON list format as before, with each dictionary representing a link between two concepts:
[
{
"node_1": "Original concept 1",
"node_2": "Original concept 2",
"edge": "Original relationship label"
},
{
"node_1": "Previously isolated concept",
"node_2": "Existing concept it connects to",
"edge": "New semantic relationship linking them"
},
{
"node_1": "Another original concept",
"node_2": "Related original concept",
"edge": "Original relationship label"
},
{
"node_1": "New bridging concept",
"node_2": "Existing concept it helps link",
"edge": "Relationship connecting distant parts of the graph"
},
{...}
]
The output should contain all of the original ontology triplets along with the newly added bridging ones, organized alphabetically by "node_1". The result will be a more interconnected and cohesive knowledge graph that captures the key concepts and relationships from the context."""
    response = str(_generate(system_prompt,prompt))
    response = response[response.find("["):response.find("]")+1]
    response = response.replace("node1","node_1")
    response = response.replace("node2","node_2")
    return response

def _fix_format(input):
    prompt = f"""given this json \nOriginal Json: {input}"""+"""

Format: ```
[
  {
    "node_1": "{response_original_concept_1}",
    "node_2": "{response_original_concept_2}", 
    "edge": "{response_original_relationship}"
  },
  {
    "node_1": "{response_previously_isolated_concept}",
    "node_2": "{response_existing_connected_concept}",
    "edge": "{response_new_semantic_relationship}"
  },
  {
    "node_1": "{response_another_original_concept}",
    "node_2": "{response_related_original_concept}",
    "edge": "{response_another_original_relationship}"
  },
  {
    "node_1": "{response_new_bridging_concept}",
    "node_2": "{response_existing_linked_concept}",
    "edge": "{response_distant_graph_relationship}"
  }
]
```

ensure that the existing JSON matching the template above. But contains all of the original ontology triplets."""
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
Context chunk:"""+sums+"""Ontologys:"""+disconnected+"""Please follow these steps to create the unified ontology:

Carefully review each context chunk to understand the key concepts, entities, and their relationships within each individual ontology.
Analyze the ontologies and their corresponding contexts to identify potential semantic relationships that could link nodes across different ontologies. Consider the following:

Look for explicit mentions of relationships between nodes from different ontologies in their respective contexts.
Infer implicit relationships based on the properties, attributes, or roles associated with nodes across ontologies.
Consider domain-specific knowledge or common sense to establish meaningful connections. Also standaridize the terminology used in the ontologies to ensure consistency.
Ensure that you provide addiotional triplets to connect all nodes in the ontology graph. Do not add comments to the JSON output.

For each identified relationship, create new triplets in the following format to represent the connections between ontologies:
{
"node_1": "Node from ontology A",
"node_2": "Node from ontology B",
"edge": "Semantic relationship"
}
Ensure that the "edge" label accurately captures the nature of the relationship between the two nodes based on the information provided in the context chunks.
If multiple relationships are identified between ontologies, create separate triplets for each relationship.
Integrate the newly created linking triplets into the individual ontology graphs to form a unified ontology.
Merge nodes that represent the same concept across different ontologies to avoid duplicates in the unified ontology. Also split long nodes into multiple nodes and edges if needed. To ensure connectivity, add intermediate nodes if necessary. 
Organize the triplets in the unified ontology alphabetically by the "node_1" field.

Please provide the unified ontology graph in the following JSON format:
[
{
"node_1": "Original concept 1",
"node_2": "Original concept 2",
"edge": "Original relationship label"
},
{
"node_1": "Previously isolated concept",
"node_2": "Existing concept it connects to",
"edge": "New semantic relationship linking them"
},
{
"node_1": "Another original concept",
"node_2": "Related original concept",
"edge": "Original relationship label"
},
{
"node_1": "New bridging concept",
"node_2": "Existing concept it helps link",
"edge": "Relationship connecting distant parts of the graph"
},
{...}
]
The output should contain all nodes from all the individual ontologies, along with the newly created edges linking them. The result will be a unified ontology graph that integrates the previously unconnected ontologies through meaningful semantic relationships based on their respective contexts."""
    response = str(_generate("",prompt))
    response = response[response.find("["):response.find("]")+1].lower()
    return response

def one_switch(ont):
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

def fix_ontology(ont, context):
    disconnected = "\n\n".join(one_switch(ont))
    prompt = f"""Here's a prompt that takes a series of unconnected ontology graphs and their corresponding context chunks, and generates a single unified ontology in JSON format that combines the individual ontologies with additional linking triplets:
Given a series of unconnected ontology graphs extracted from their respective context chunks, your task is to analyze the contexts and identify potential relationships that could link these isolated ontologies to form a single, cohesive knowledge graph. The goal is to create a unified ontology where all the individual ontologies are connected through meaningful semantic relationships.
Please provide the context chunks and their corresponding ontologies in the following format:
Context chunk:"""+context+"""Ontologys:"""+disconnected+"""Please follow these steps to create the unified ontology:

Carefully review each context chunk to understand the key concepts, entities, and their relationships within each individual ontology.
Analyze the ontologies and their corresponding contexts to identify potential semantic relationships that could link nodes across different ontologies. Consider the following:

Look for explicit mentions of relationships between nodes from different ontologies in their respective contexts.
Infer implicit relationships based on the properties, attributes, or roles associated with nodes across ontologies.
Consider domain-specific knowledge or common sense to establish meaningful connections. Also standaridize the terminology used in the ontologies to ensure consistency.
Ensure that you provide addiotional triplets to connect all nodes in the ontology graph. Do not add comments to the JSON output.

For each identified relationship, create new triplets in the following format to represent the connections between ontologies:
{
"node_1": "Node from ontology A",
"node_2": "Node from ontology B",
"edge": "Semantic relationship"
}
Ensure that the "edge" label accurately captures the nature of the relationship between the two nodes based on the information provided in the context chunks.
If multiple relationships are identified between ontologies, create separate triplets for each relationship.
Integrate the newly created linking triplets into the individual ontology graphs to form a unified ontology.
Merge nodes that represent the same concept across different ontologies to avoid duplicates in the unified ontology. Also split long nodes into multiple nodes and edges if needed. To ensure connectivity, add intermediate nodes if necessary. 
Organize the triplets in the unified ontology alphabetically by the "node_1" field.

Please provide the unified ontology graph in the following JSON format:
[
{
"node_1": "Original concept 1",
"node_2": "Original concept 2",
"edge": "Original relationship label"
},
{
"node_1": "Previously isolated concept",
"node_2": "Existing concept it connects to",
"edge": "New semantic relationship linking them"
},
{
"node_1": "Another original concept",
"node_2": "Related original concept",
"edge": "Original relationship label"
},
{
"node_1": "New bridging concept",
"node_2": "Existing concept it helps link",
"edge": "Relationship connecting distant parts of the graph"
},
{...}
]
The output should contain all nodes from all the individual ontologies, along with the newly created edges linking them. The result will be a unified ontology graph that integrates the previously unconnected ontologies through meaningful semantic relationships based on their respective contexts."""
    response = str(_generate("",prompt))
    response = response[response.find("["):response.find("]")+1].lower()
    return response


import networkx as nx
def new_summary_prompt(summary, text_chunk):
    summary_prompt = f"""Given an existing summary and a new chunk of text containing additional information, your task is to update the summary to seamlessly integrate the relevant details from the new text. The goal is to produce an enhanced summary that incorporates the most salient points from both the original summary and the new text chunk.
Please follow these steps:

Carefully read the existing summary to understand the key points and overall context.

Identify specific details, facts, or concepts in the new text chunk that provide additional context, elaborate on key points, or introduce new relevant information not covered in the original summary.
Revise the existing summary to incorporate the identified information from the new text chunk:

Add new sentences or phrases to introduce additional details or context where appropriate.
Modify existing sentences to accommodate the new information, ensuring the summary remains coherent and logical.
If necessary, remove or condense less critical points from the original summary to maintain a concise and focused presentation.


Ensure that the updated summary integrates the new information seamlessly, maintaining a coherent narrative flow and consistent style.
Review the revised summary to check for clarity, coherence, and logical consistency. Make any necessary adjustments to improve readability and effectiveness.

Please provide the updated summary that incorporates the relevant information from the new text chunk while preserving the most important points from the original summary.
Updated summary:"""
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
            combinations[0] = fix_ontology(combinations[0], summaries[0])
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
        system_prompt="""You are a question and answer system that uses a knowledge graph as input to provide concise, direct answers to user queries based on the information in the knowledge graph. You do not mention or reference the fact that you were given material to work with. Instead, you simply provide the most relevant and accurate answer to the question.
For example, if the question is "What was the first dog to land on the moon?", you return the answer: "No dog has ever landed on the moon." without any additional context or explanation about the knowledge graph.
You handle a wide range of questions and provide answers based on the information available in the knowledge graph. If the answer cannot be found within the provided context, you respond with "I do not have enough information to answer that question."
Your focus is on delivering clear, concise answers without any unnecessary information or references to the underlying knowledge graph.""",
    )
    response = chat_engine.chat(
        "Do not start the response with any preface and remember to answer as if you've always known the answer not that you where provided with a context \n\nprompt:"+prompt,
    )
    return response.response

def create_KG_from_text(text, output_file="./output/"):
    jsons = _create_kg(_get_text_chunks(text), converege=False, repeats=1)
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
    text = _url_to_md(url)
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
