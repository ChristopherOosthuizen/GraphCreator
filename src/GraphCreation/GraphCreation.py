# Created by Christopher Oosthuizen on 06/22/2024
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
from openai import OpenAI
import threading
import networkx as nx
import pandas as pd
from pyvis.network import Network
from pdfminer.high_level import extract_text
import textformatting
import json
import LinkPrediction as lp
import LLMFunctions as LLM
from cdlib import algorithms 
import random
from flair.data import Sentence
from flair.nn import Classifier
tagger = None
def create_knowledge_triplets(text_chunk="", repeats=5, ner=False, model_id=0, ner_type="flair"):
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
            sentence = sentence.replace(text_chunk,"")
            system_prompt = open("../prompts/NERTripletCreation").read()+sentence
        else:
            system_prompt = open("../prompts/NERTripletCreation").read()+ LLM.generate_chat_response(open("../prompts/NERprompt").read(), text_chunk, model_id=model_id)
    else:
        system_prompt = open("../prompts/TripletCreationSystem").read()
    prompt = f"Context: ```{text_chunk}``` \n\nOutput: "
    response = str(LLM.generate_chat_response(system_prompt, prompt, model_id=model_id))
    
    for _ in range(repeats):
        system_prompt = open("../prompts/TripletCreationSystem").read()
        prompt = f"""Here is the prompt updated to insert additional triplets into the existing ontology:
Read this context carefully and extract the key concepts and relationships discussed:
{text_chunk}
Here is the ontology graph generated from the above context:
{response}
{open("../prompts/TripletIterationStandard").read()}"""
        response = str(LLM.generate_chat_response(system_prompt, prompt, model_id=model_id))
    
    response = str(lp._fix_ontology(response, text_chunk, model_id=model_id))
    response = response[response.find("["):response.find("]")+1]
    response = response.replace("node1", "node_1")
    response = response.replace("node2", "node_2")
    
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
    summary_prompt = open("../prompts/summaryPrompt").read()
    summary = LLM.generate_chat_response(summary_prompt, f"existing_summary: {summary} new_text_chunk: {text_chunk}", model_id=model_id)
    return summary

def _make_one_triplet(list, position, chunk, ner=False, ner_type="flair"):
    chu = create_knowledge_triplets(text_chunk=chunk, model_id=LLM.pick_gpu(position), ner=ner, ner_type=ner_type)
    list[position] = chu

def _combine_one(ont1, ont2, sum1, sum2, list, position, summaries):
    model_id = LLM.pick_gpu(position)
    sums = new_summary_prompt(sum1, sum2, model_id=model_id)
    summaries[position] = sums
    list[position] = lp._combine_ontologies(ont1, ont2, sums, model_id=model_id)

    

def _create_kg(chunks, repeats=5, converge=True, inital_repeats=2, ner=False, ner_type="flair"):
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
        return [create_knowledge_triplets(chunks[0], repeats=inital_repeats, ner=ner,ner_type=ner_type)]
    combinations = []
    summaries = []
    threads = []
    triplets = [""]*len(chunks)
    for x in range(len(chunks)):
        thread = threading.Thread(target=_make_one_triplet, args=(triplets,x,chunks[x],ner,ner_type))
        threads.append(thread)
        thread.start()
    for thread in threads:
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
            thread.join()
        if len(combinations)%2 == 1:
            combinations[-1] = old_combinations[-1]
            summaries[-1] = old_summaries[-1]
    if converge:
        while len(lp.ontologies_to_unconncected(combinations[0], combinations[0])) > 1:
            combinations[0] = lp._fix_ontology(combinations[0], summaries[0])
        return combinations
    for x in range(len(combinations)-1, 0, -1):
        if combinations[x].strip() == "":
            combinations.pop(x)
    return combinations


def create_KG_from_text(text, output_file="./output/", eliminate_all_islands=False, inital_repeats=2, chunks_precentage_linked=0.5, llm_formatting=True, ner=False, ner_type="flair"):
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
    return create_KG_from_chunks(chunks, output_file, eliminate_all_islands, inital_repeats, chunks_precentage_linked, ner, ner_type)

def create_KG_from_chunks(chunks, output_file="./output/", eliminate_all_islands=False, inital_repeats=2, chunks_precentage_linked=0.5, ner=False, ner_type="flair"):
    repeats = int(chunks_precentage_linked * len(chunks))
    jsons = _create_kg(chunks=chunks, converge=eliminate_all_islands, repeats=repeats, inital_repeats=inital_repeats, ner=ner, ner_type=ner_type)
    Graph = nx.Graph()
    if not os.path.exists(output_file):
        os.makedirs(output_file)
    for x in jsons:
        try:
            x = json.loads(x)
        except:
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
    
    return chunks,Graph
def create_KG_from_url(url, output_file="./output/", eliminate_all_islands=False, inital_repeats=2, chunks_precentage_linked=0.5,llm_formatting=True, ner=False, ner_type="flair"):
    text = textformatting.url_to_md(url)
    jsons = create_KG_from_text(text, output_file, eliminate_all_islands,inital_repeats, chunks_precentage_linked, llm_formatting,ner, ner_type)
    return jsons

def create_KG_from_pdf(pdf, output_file="./output/", eliminate_all_islands=False, inital_repeats=2, chunks_precentage_linked=0.5,llm_formatting=True, ner=False, ner_type="flair"):
    text = textformatting._convert_to_markdown(extract_text(pdf))
    jsons = create_KG_from_text(text, output_file, eliminate_all_islands, inital_repeats, chunks_precentage_linked, llm_formatting, ner, ner_type)
    return jsons