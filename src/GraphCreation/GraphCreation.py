# Created by Christopher Oosthuizen on 06/22/2024
from openai import OpenAI
import threading
import networkx as nx
import pandas as pd
from pyvis.network import Network
from pdfminer.high_level import extract_text
from llama_index.core import set_global_tokenizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from llama_index.core import StorageContext
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import Settings
from llama_index.core.chat_engine import ContextChatEngine
import textformatting
import json
import LinkPrediction as lp
from concurrent.futures import ThreadPoolExecutor


def create_knowledge_triplets(text_chunk="", repeats=5):
    """
    Creates knowledge triplets from a given text chunk.

    Args:
        text_chunk (str): The text chunk to extract knowledge triplets from.
        repeats (int): The number of times to repeat the process.

    Returns:
        str: The generated knowledge triplets in JSON format.
    """
    system_prompt = open("prompts/TripletCreationSystem").read()
    prompt = f"Context: ```{text_chunk}``` \n\nOutput: "
    response = str(lp.generate_chat_response(system_prompt, prompt))
    
    for _ in range(repeats):
        system_prompt = open("prompts/TripletCreationSystem").read()
        prompt = f"""Here is the prompt updated to insert additional triplets into the existing ontology:
Read this context carefully and extract the key concepts and relationships discussed:
{text_chunk}
Here is the ontology graph generated from the above context:
{response}
{open("prompts/TripletIterationStandard").read()}"""
        response = str(lp.generate_chat_response(system_prompt, prompt))
    
    response = str(lp.fix_ontology(response))
    response = response[response.find("["):response.find("]")+1]
    response = response.replace("node1", "node_1")
    response = response.replace("node2", "node_2")
    
    return response



def new_summary_prompt(summary, text_chunk):
    """
    Generates a new summary using the given summary and text chunk.

    Args:
        summary (str): The existing summary.
        text_chunk (str): The new text chunk to be added to the summary.

    Returns:
        str: The generated summary.
    """
    summary_prompt = open("prompts/summaryPrompt").read()
    summary = lp.generate_chat_response(summary_prompt, f"existing_summary: {summary} new_text_chunk: {text_chunk}")
    return summary

def _make_one_triplet(list, position, chunk):
    list[position] = create_knowledge_triplets(text_chunk=chunk)

def _combine_one(ont1, ont2, sum1, sum2, list, position, summaries):
    sums = new_summary_prompt(sum1, sum2)
    summaries[position] = sums
    list[position] = lp._combine_ontologies(ont1, ont2, sums)

def _create_kg(chunks, repeats=5, converge=True):
    """
    Creates a knowledge graph from a list of text chunks.

    Args:
        chunks (list): A list of text chunks.
        repeats (int, optional): The number of times to repeat the combination process. Defaults to 5.
        converge (bool, optional): Whether to converge the knowledge graph by fixing ontologies. Defaults to True.

    Returns:
        list: The generated knowledge graph.

    """
    print(f"Number of chunks: {len(chunks)}")
    triplets = {}
    combinations = {}
    if len(chunks) == 1:
        return [create_knowledge_triplets(text_chunk=chunks[0])]
    
    with ThreadPoolExecutor() as executor:
        for i, chunk in enumerate(chunks):
            executor.submit(_make_one_triplet, triplets, i, chunk)
    
    combinations = triplets
    summaries = chunks
    
    for _ in range(repeats):
        if len(combinations) == 1:
            break
        
        old_combinations = combinations
        old_summaries = summaries
        summaries = {}
        combinations = {}
        
        with ThreadPoolExecutor() as executor:
            for i in range(1, len(old_combinations), 2):
                executor.submit(_combine_one, old_combinations[i - 1], old_combinations[i], old_summaries[i - 1], old_summaries[i], combinations, i // 2, summaries)
        
        if len(old_combinations) % 2 == 1:
            combinations[len(old_combinations) // 2] = old_combinations[len(old_combinations) - 1]
            summaries[len(old_combinations) // 2] = old_summaries[len(old_combinations) - 1]
    
    if converge:
        while len(lp._ontologies_to_unconnected(combinations[0], combinations[0])) > 1:
            print(len(lp._ontologies_to_unconnected(combinations[0], combinations[0])))
            combinations[0] = lp._fix_ontology(combinations[0], summaries[0])
    
    return list(combinations.values())

def graphquestions(graph, prompt):
    """
    Function to ask questions about a graph.

    Args:
        graph (Graph): The graph to ask questions about.
        prompt (str): The question prompt.

    Returns:
        str: The response to the question.
    """
    graph_store = SimpleGraphStore()
    for node_1, node_2, data in graph.edges(data=True):
        graph_store.upsert_triplet(node_1, data['title'], node_2)
    
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    graph_rag_retriever = KnowledgeGraphRAGRetriever(
        storage_context=storage_context,
        verbose=False,
    )
    chat_engine = ContextChatEngine.from_defaults(
        retriever=graph_rag_retriever,
        verbose=False,
        max_entities=30,
        graph_traversal_depth=10,
        max_knowledge_sequence=15,
    )
    
    response = chat_engine.chat(prompt)
    return response.response

def create_KG_from_text(text, output_file="./output/"):
    """
    Creates a knowledge graph (KG) from the given text.

    Parameters:
    text (str): The input text from which the KG will be created.
    output_file (str): The path to the output file directory. Default is "./output/".

    Returns:
    nx.Graph: The created knowledge graph.

    """
    jsons = _create_kg(textformatting._get_text_chunks(text), converge=False, repeats=1)
    Graph = nx.Graph()
    for x in jsons:
        try:
            x = json.loads(x)
        except:
            print(jsons)
            x = json.loads(lp.fix_format(x))
        for y in x:
            Graph.add_edge(y["node_1"], y["node_2"], label=y["edge"])
    
    # Save graph as GraphML
    nx.write_graphml(Graph, output_file + "graph.graphml")
    
    # Save graph as JSON
    with open(output_file + "graph.json", "w") as json_file:
        json.dump(jsons, json_file)
    
    # Draw and save graph as HTML
    nt = Network('1000px', '1000px', notebook=False)
    nt.from_nx(Graph)
    nt.show(output_file + "graph.html", notebook=False)
    
    return Graph

def create_KG_from_url(url, output_file="./output/"):
    text = textformatting._url_to_md(url)
    jsons = create_KG_from_text(text, output_file)
    return jsons

def create_KG_from_pdf(pdf, output_file="./output/"):
    text = textformatting._convert_to_markdown(extract_text(pdf))
    print(text)
    jsons = create_KG_from_text(text, output_file)
    return jsons
