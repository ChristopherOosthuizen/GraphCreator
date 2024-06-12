# Created by Christopher Oosthuizen on 06/22/2024
import os
from openai import OpenAI
import threading
import networkx as nx
import pandas as pd
from pyvis.network import Network
from pdfminer.high_level import extract_text
import json
from cdlib import algorithms 
import random
from flair.data import Sentence
from flair.nn import Classifier
import math
from . import textformatting
from . import LinkPrediction as lp
from . import LLMFunctions as LLM
from cdlib import algorithms
import random
import re
import colorsys
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
prompts_dir = os.path.join(current_dir, '..', 'prompts')
tagger = None
def create_knowledge_triplets(text_chunk="", repeats=5, ner=False, model_id=0, ner_type="flair", num=10):
    """
    Creates knowledge triplets from a given text chunk.

    Args:
        text_chunk (str): The text chunk to extract knowledge triplets from.
        repeats (int): The number of times to repeat the process.

    Returns:
        str: The generated knowledge triplets in JSON format.
    """
    system_prompt = ""
    if ner:
        if ner_type == "flair":
            global tagger
            if tagger is None:
                tagger = Classifier.load("ner-ontonotes-fast")
            sentence= Sentence(text_chunk)
            tagger.predict(sentence)
            sentence = str(sentence).replace(text_chunk,"")
            system_prompt = open(os.path.join(prompts_dir,"NERTripletCreation")).read().replace("<num>",str(num))+sentence
        else:
            system_prompt = open(os.path.join(prompts_dir,"NERTripletCreation")).read()+ LLM.generate_chat_response(open(os.path.join(prompts_dir,"NERprompt")).read().replace("<num>",str(num)), text_chunk, model_id=model_id)
    else:
        system_prompt = open(os.path.join(prompts_dir,"TripletCreationSystem")).read().replace("<num>",str(num))
    prompt = f"Context: ```{text_chunk}``` \n\nOutput: "
    response = str(LLM.generate_chat_response(system_prompt, prompt, model_id=model_id))
    response = response[response.find("["):response.find("]")+1]
    response = response.replace("node1", "node_1")
    response = response.replace("node2", "node_2")
    times = 0
    if repeats != 0:
        while(LLM.generate_chat_response("", open(os.path.join(prompts_dir,"infer")).read().replace("<context>",text_chunk).replace("<triplets>",response), model_id=model_id) == "yes" and times < repeats):
            system_prompt = open(os.path.join(prompts_dir,"TripletCreationSystem")).read()
            prompt = f"Context Chunk: {text_chunk} Ontology: {response} \n\nOutput: "
            new_edges = str(LLM.generate_chat_response(system_prompt, prompt, model_id=model_id))
            new_edges = new_edges[new_edges.find("["):new_edges.find("]")+1]
            new_edges = new_edges.replace("node1", "node_1")
            new_edges = new_edges.replace("node2", "node_2")
            new_edges = new_edges.replace("\'", "\"")
            new_edges = new_edges.replace("}\n", "},\n")
            new_edges = re.sub('^{', '[\n{', new_edges)
            new_edges = re.sub('}$', '}\n]', new_edges)
            new_edges = re.sub('},\n]', '}\n]', new_edges)
            response = "["+",\n".join(str(x) for x in (json.loads(response) +json.loads(new_edges)))+"]"
            response = response[response.find("["):response.find("]")+1]
            response = response.replace("node1", "node_1")
            response = response.replace("node2", "node_2")
            response = response.replace("\'", "\"")
            response = response.replace("}\n", "},\n")
            response = re.sub('^{', '[\n{', response)
            response = re.sub('}$', '}\n]', response)
            response = re.sub('},\n]', '}\n]', response)
            times += 1
        
    
    response = str(lp._fix_ontology(response, text_chunk, model_id=model_id))
    response = response[response.find("["):response.find("]")+1]
    response = response.replace("node1", "node_1")
    response = response.replace("node2", "node_2")
    return lp.strip_json(response)
    



def new_summary_prompt(summary, text_chunk,model_id=0):
    """
    Generates a new summary using the given summary and text chunk.

    Args:
        summary (str): The existing summary.
        text_chunk (str): The new text chunk to be added to the summary.

    Returns:
        str: The generated summary.
    """
    summary_prompt = open(os.path.join(prompts_dir,"summaryPrompt")).read()
    summary = LLM.generate_chat_response(summary_prompt, f"summary1: {summary}\n\n summary2: {text_chunk}", model_id=model_id)
    return summary

def _make_one_triplet(list, position, chunk, ner=False, ner_type="flair",num=10):
    chu = create_knowledge_triplets(text_chunk=chunk, model_id=LLM.pick_gpu(position), ner=ner, ner_type=ner_type, num=num)
    list[position] = chu

def _combine_one(ont1, ont2, sum1, sum2, list, position, summaries):
    model_id = LLM.pick_gpu(position)
    sums = new_summary_prompt(sum1, sum2, model_id=model_id)
    summaries[position] = sums
    list[position] = lp._combine_ontologies(ont1, ont2, sums, model_id=model_id)

    
def _converge_lists(lists, summaries, repeats=.5):
    combinations = lists
    length = len(lists)
    num_iterations = math.ceil(math.log2(length) * repeats)
    for _ in range(num_iterations):
        if len(combinations) == 1:
            break
        old_combinations = combinations
        old_summaries = summaries
        summaries = [""] * ((len(combinations) // 2) + (len(combinations) % 2))
        combinations = [""] * ((len(combinations) // 2) + (len(combinations) % 2))
        threads = []
        for x in range(1, len(old_combinations), 2):
            thread = threading.Thread(target=_combine_one, args=(old_combinations[x - 1], old_combinations[x], old_summaries[x - 1], old_summaries[x], combinations, x // 2, summaries))
            threads.append(thread)
            thread.start()
        if len(old_combinations) % 2 != 0:
            combinations[-1] = old_combinations[-1]
            summaries[-1] = old_summaries[-1]
        for thread in threads:
            thread.join()
    return combinations, summaries
def _create_kg(chunks, repeats=.5, converge=True, inital_repeats=2, ner=False, ner_type="flair",num=10):
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
    triplets = []
    combinations = [] 
    if len(chunks) == 1:
        return [create_knowledge_triplets(chunks[0], repeats=inital_repeats, ner=ner,ner_type=ner_type,num=num)]
    combinations = []
    summaries = []
    threads = []
    triplets = [""]*len(chunks)
    for x in range(len(chunks)):
        thread = threading.Thread(target=_make_one_triplet, args=(triplets,x,chunks[x],ner,ner_type,num))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    combinations = triplets
    summaries = chunks
    new_combinations = [] 
    new_summaries = []
    for x in range(len(combinations)):
        if combinations[x].strip() != "":
            new_combinations.append(combinations[x])
            new_summaries.append(summaries[x])
    combinations = new_combinations
    summaries = new_summaries
    combinations, summaries = _converge_lists(combinations, summaries, repeats=repeats)
    if converge:
        while len(lp._ontologies_to_unconnected(combinations[0], combinations[0])) > 1:
            print(len(lp._ontologies_to_unconnected(combinations[0], combinations[0])))
            combinations[0] = lp._fix_ontology(combinations[0], summaries[0])
    new_combinations = [] 
    new_summaries = []
    for x in range(len(combinations)):
        if combinations[x].strip() != "":
            new_combinations.append(combinations[x])
            new_summaries.append(summaries[x])
    combinations = new_combinations
    summaries = new_summaries
    return combinations


def create_KG_from_text(text, output_file="./output/", eliminate_all_islands=False, inital_repeats=2, chunks_precentage_linked=0.5, llm_formatting=True, ner=False, ner_type="flair",num=10):
    """
    Creates a knowledge graph (KG) from the given text.

    Parameters:
    text (str): The input text from which the KG will be created.
    output_file (str): The path to the output file directory. Default is "./output/".

    Returns:
    nx.Graph: The created knowledge graph.

    """
    chunks = []
    if llm_formatting:
        chunks = textformatting.get_text_chunks(text)
    else:
        chunks = textformatting.chunk_text(text)
    return create_KG_from_chunks(chunks, output_file, eliminate_all_islands, inital_repeats, chunks_precentage_linked, ner, ner_type,num)

    # Assign colors to nodes based on clusters
def generate_colors(num_clusters):
    colors = []
    for i in range(num_clusters):
        hue = i / num_clusters
        saturation = 0.5 + (i % 2) * 0.2
        value = 0.8 - (i % 3) * 0.1
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        color = "#{:02x}{:02x}{:02x}".format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
        colors.append(color)
    return colors
    

def create_KG_from_chunks(chunks, output_file="./output/", eliminate_all_islands=False, inital_repeats=2, chunks_precentage_linked=0.5, ner=False, ner_type="flair",num=10):
    if not os.path.exists(output_file):
        os.makedirs(output_file)
    repeats = chunks_precentage_linked
    jsons = _create_kg(chunks=chunks, converge=eliminate_all_islands, repeats=repeats, inital_repeats=inital_repeats, ner=ner, ner_type=ner_type,num=num)
    Graph = nx.Graph()
    for x in jsons:
        try:

            x = json.loads(x)
        except Exception as err:
            x = json.loads(lp.fix_format(x,str(err)))
        
        for y in x:
            Graph.add_node(y["node_1"],label=y["node_1"])
            Graph.add_node(y["node_2"],label=y["node_2"])
            Graph.add_edge(y["node_1"], y["node_2"], label=y["edge"])
    
    # Save graph as GraphML
    nx.write_graphml(Graph, output_file + "/graph.graphml")
    
    # Save graph as JSON
    with open(output_file + "/graph.json", "w") as json_file:
        json.dump(jsons, json_file)
    
    # Draw and save graph as HTML
    nt = Network(height="1000px",width="100%",bgcolor="#222222",font_color="white", notebook=False)
    nt.from_nx(Graph)
    nt.show(output_file + "/graph.html", notebook=False)
    
    # Cluster nodes using Leiden algorithm
    leiden_communities = algorithms.leiden(Graph)

    num_clusters = len(leiden_communities.communities)
    colors = generate_colors(num_clusters)
    for i, community in enumerate(leiden_communities.communities):
        for o, node in enumerate(community):
            nt.get_node(node)['color'] = colors[i % num_clusters]
    # Save clustered graph as HTML
    nt.show(output_file + "/clustered_graph.html", notebook=False)
    
    return chunks, Graph
def create_KG_from_url(url, output_file="./output/", eliminate_all_islands=False, inital_repeats=2, chunks_precentage_linked=0.5,llm_formatting=True, ner=False, ner_type="flair",num=10):
    text = textformatting.url_to_md(url)
    jsons = create_KG_from_text(text, output_file, eliminate_all_islands,inital_repeats, chunks_precentage_linked, llm_formatting,ner, ner_type,num)
    return jsons
def create_KG_from_pdf(pdf, output_file="./output/", eliminate_all_islands=False, inital_repeats=2, chunks_precentage_linked=0.5,llm_formatting=True, ner=False, ner_type="flair",num=10):
    text = textformatting.pdf_to_md(pdf)
    jsons = create_KG_from_text(text, output_file, eliminate_all_islands, inital_repeats, chunks_precentage_linked, llm_formatting, ner, ner_type,num)
    return jsons

def create_KG_from_folder(folder, output_file="./output/", eliminate_all_islands=False, inital_repeats=2, chunks_precentage_linked=0.5,llm_formatting=True, ner=False, ner_type="flair",num=10):
    files = os.listdir(folder)
    text = ""
    for file in files:
        if file.endswith(".pdf"):
            create_KG_from_pdf(folder + file, output_file, eliminate_all_islands, inital_repeats, chunks_precentage_linked, llm_formatting, ner, ner_type,num)
        elif file.endswith(".html"):
            create_KG_from_url(folder + file, output_file, eliminate_all_islands, inital_repeats, chunks_precentage_linked, llm_formatting, ner, ner_type,num)
        else:
            with open(folder + file, "r") as f:
                text = f.read()
                create_KG_from_text(text, output_file, eliminate_all_islands, inital_repeats, chunks_precentage_linked, llm_formatting, ner, ner_type,num)