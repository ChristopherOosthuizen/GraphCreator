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
    for x in range(repeats):
        system_prompt = open("prompts/TripletCreationSystem").read()
        prompt = """Here is the prompt updated to insert additional triplets into the existing ontology:
Read this context carefully and extract the key concepts and relationships discussed:""" + text_chunk + """Here is the ontology graph generated from the above context:""" + response + open("prompts/TripletIterationStandard").read()
        response = str(lp.generate_chat_response(system_prompt, prompt))
    response = str(lp.fix_ontology(response))
    response = response[response.find("["):response.find("]")+1]
    response = response.replace("node1", "node_1")
    response = response.replace("node2", "node_2")
    return response



def new_summary_prompt(summary, text_chunk):
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
    print(f"Number of chunks: {len(chunks)}")
    triplets = []
    combinations = []
    if len(chunks) == 1:
        return [create_knowledge_triplets(text_chunk=chunks[0])]
    combinations = []
    summaries = []
    threads = []
    triplets = [""] * len(chunks)
    for x in range(len(chunks)):
        thread = threading.Thread(target=_make_one_triplet, args=(triplets, x, chunks[x]))
        threads.append(thread)
        thread.start()
    for thread in threads:
        print(x / len(chunks))
        thread.join()
    combinations = triplets
    summaries = chunks
    for x in range(repeats):
        if len(combinations) == 1:
            break
        old_combinations = combinations
        old_summaries = summaries
        summaries = [""] * (int(len(combinations) / 2) + len(combinations) % 2)
        combinations = [""] * (int(len(combinations) / 2) + len(combinations) % 2)
        threads = []
        for x in range(1, len(old_combinations), 2):
            thread = threading.Thread(target=_combine_one, args=(old_combinations[x - 1], old_combinations[x], old_summaries[x - 1], old_summaries[x], combinations, x // 2, summaries))
            threads.append(thread)
            thread.start()
        for thread in threads:
            print(1 / len(combinations))
            thread.join()
        if len(combinations) % 2 == 1:
            combinations[-1] = old_combinations[-1]
            summaries[-1] = old_summaries[-1]
    if converge:
        while len(lp._ontologies_to_unconnected(combinations[0], combinations[0])) > 1:
            print(len(lp._ontologies_to_unconnected(combinations[0], combinations[0])))
            combinations[0] = lp._fix_ontology(combinations[0], summaries[0])
        return combinations
    return combinations

def graphquestions(graph, prompt):
    graph_store = SimpleGraphStore()
    for x in graph.edges(data=True):
        graph_store.upsert_triplet(x[0], x[2]['title'], x[1])
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
    nx.write_graphml(Graph, output_file + "graph.graphml")
    open(output_file + "graph.json", "w").write(json.dumps(jsons))
    nx.draw(Graph, with_labels=True)
    nt = Network('100%', '100%')
    nt.from_nx(Graph)
    nt.show(output_file + "graph.html", notebook=False)
    return Graph

def create_KG_from_url(url, output_file="./output/"):
    text = textformatting._url_to_md(url)
    jsons = create_KG_from_text(text, output_file)
    return jsons

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
