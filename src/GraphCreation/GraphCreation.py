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
from . import BenchMarks as bm
from cdlib import algorithms
import random
import re
import colorsys
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
prompts_dir = os.path.join(current_dir,'prompts')
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
    system_prompt = open(os.path.join(prompts_dir,"TripletCreationSystem")).read().replace("<num>",str(num))
    shorthand_list = LLM.generate_chat_response(open(os.path.join(prompts_dir,"NERprompt")).read(), text_chunk, model_id=model_id).split("\n")
    shorthands = textformatting.token_compression(shorthand_list)
    shorthand_dict = shorthands
    shorthands = [x+":"+shorthands[x] for x in shorthands.keys()]
    shorthands = "\n".join(shorthands)
    system_prompt = system_prompt.replace("<shorthands>",shorthands)

    prompt = f"Context:{text_chunk}\n\nOutput: "
    response = str(LLM.generate_chat_response(system_prompt, prompt, model_id=model_id)).replace("```","").strip()
    times = 0
    if repeats != 0:
        while(LLM.generate_chat_response("", open(os.path.join(prompts_dir,"infer")).read().replace("<context>",text_chunk).replace("<triplets>",response), model_id=model_id) == "yes" and times < repeats):
            system_prompt = open(os.path.join(prompts_dir,"TripletIterationStandard")).read()
            system_prompt = system_prompt.replace("<shorthands>",shorthands)
            prompt = f"Context Chunk: {text_chunk} Ontology: {response} \n\nOutput: "
            new_edges = str(LLM.generate_chat_response(system_prompt, prompt, model_id=model_id)).replace("```","").strip()
            unrolled = textformatting.unroll_triplets(new_edges).split("\n")
            shorthand_dict = textformatting.expand_compress(unrolled,shorthand_dict)
            shorthands = [x+":"+shorthand_dict[x] for x in shorthand_dict.keys()]
            shorthands = "\n".join(shorthands)
            response = response+new_edges
            times += 1
    response = textformatting.decompress(response,shorthand_dict).strip()
    return response
    



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
    old_combinations = combinations
    old_summaries = summaries
    summaries = [""] * (len(combinations)-1)
    combinations = [""] *(len(combinations)-1)
    threads = []
    for x in range(1, len(old_combinations)):
        thread = threading.Thread(target=_combine_one, args=(old_combinations[x - 1], old_combinations[x], old_summaries[x - 1], old_summaries[x], combinations, x-1, summaries))
        threads.append(thread)
        thread.start()
    if len(old_combinations) % 2 != 0:
        combinations[-1] = old_combinations[-1]
        summaries[-1] = old_summaries[-1]
    for thread in threads:
        thread.join()
    return combinations, summaries
def threadsOptimized(triplets,x,chunks,ner,ner_type,num):
    threads = []
    for x in range(len(chunks)):
        if x%5 == 0:
            for thread in threads:
                thread.join()
            threads = []
        thread = threading.Thread(target=_make_one_triplet, args=(triplets,x,chunks[x],ner,ner_type,num))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
def threadsFull(triplets,x,chunks,ner,ner_type,num):
    threads = []
    for x in range(len(chunks)):
        thread = threading.Thread(target=_make_one_triplet, args=(triplets,x,chunks[x],ner,ner_type,num))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
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
    if os.environ.get("HF_HOME") is not None:
        threadsOptimized(triplets,0,chunks,ner,ner_type,num)
    else:
        threadsFull(triplets,0,chunks,ner,ner_type,num)
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


def create_KG_from_text(text, output_file="./output/", eliminate_all_islands=False, inital_repeats=30, chunks_precentage_linked=0, llm_formatting=False, ner=False, ner_type="flair",num=5,compression=0.33,additional=""):
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
    return create_KG_from_chunks(chunks, output_file, eliminate_all_islands, inital_repeats, chunks_precentage_linked, ner, ner_type,num, compression, additional)

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
    

def create_KG_from_chunks(chunks, output_file="./output/", eliminate_all_islands=False, inital_repeats=30, chunks_precentage_linked=0, ner=False, ner_type="flair",num=5, compression=0.33, additional=""):
    if not os.path.exists(output_file):
        os.makedirs(output_file)
    repeats = chunks_precentage_linked
    jsons = _create_kg(chunks=chunks, converge=eliminate_all_islands, repeats=repeats, inital_repeats=inital_repeats, ner=ner, ner_type=ner_type,num=num)
    jsons.append(additional)
    Graph = nx.Graph()
    for x in jsons:
        splits = x.split("\n")
        for y in splits:
            objects = y.split(",")
            if len(objects) < 3:
                print(objects)
                continue
            Graph.add_node(objects[0],label=objects[0])
            Graph.add_node(objects[2],label=objects[2])
            Graph.add_edge(objects[0],objects[2], label=objects[1])
    
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
    try:
        leiden_communities = algorithms.leiden(Graph)

        num_clusters = len(leiden_communities.communities)
        colors = generate_colors(num_clusters)
        for i, community in enumerate(leiden_communities.communities):
            for o, node in enumerate(community):
                nt.get_node(node)['color'] = colors[i % num_clusters]
        # Save clustered graph as HTML
        nt.show(output_file + "/clustered_graph.html", notebook=False)
    except:
        print("Could not cluster graph.")
    return chunks, Graph
def create_KG_from_url(url, output_file="./output/", eliminate_all_islands=False, inital_repeats=30, chunks_precentage_linked=0,llm_formatting=False, ner=False, ner_type="llm",num=5, compression=0.33):
    text = textformatting.url_to_md(url,compression=compression)
    table_data = textformatting.get_tables_from_url(url)
    additional = ""
    for table in table_data:
        additional += textformatting.get_triplets_from_table(table)
    jsons = create_KG_from_text(text, output_file, eliminate_all_islands,inital_repeats, chunks_precentage_linked, llm_formatting,ner, ner_type,num,compression=compression,additional=additional)

    return jsons
def create_KG_from_pdf(pdf, output_file="./output/", eliminate_all_islands=False, inital_repeats=30, chunks_precentage_linked=0,llm_formatting=False, ner=False, ner_type="llm",num=5,compression=0.33):
    text = textformatting.pdf_to_md(pdf,compression=compression)
    jsons = create_KG_from_text(text, output_file, eliminate_all_islands, inital_repeats, chunks_precentage_linked, llm_formatting, ner, ner_type,num,compression)
    return jsons

def create_KG_from_folder(folder, output_file="./output/", eliminate_all_islands=False, inital_repeats=30, chunks_precentage_linked=0,llm_formatting=False, ner=False, ner_type="flair",num=5,compression=0.33):
    text = textformatting.folder_to_md(folder,compression=compression)
    jsons = create_KG_from_text(text, output_file, eliminate_all_islands, inital_repeats, chunks_precentage_linked, llm_formatting, ner, ner_type,num,compression)
    return jsons