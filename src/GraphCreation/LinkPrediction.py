import json
import pandas as pd
import networkx as nx

import os 
import LLMFunctions as LLM

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)



def fix_format(input, model_id=0):
    """
    Fixes the format of the input JSON.

    Args:
        input (str): The input JSON to fix the format of.

    Returns:
        str: The fixed format of the input JSON.
    """
    prompt = f"given this json \nOriginal Json: {input}" + open("../prompts/TripletCreationSystem").read()
    response = str(LLM.generate_chat_response("", prompt, model_id=model_id))
    response = response[response.find("["):response.find("]")+1]
    return response

def _triplets_to_json(triplets):
    df = pd.DataFrame({"node_1": [], "node_2": [], "edge": []})
    for triplet in triplets:
        df = df._append({"node_1": triplet[0], "node_2": triplet[2], "edge": triplet[1]}, ignore_index=True)
    return json.dumps(df.to_dict(orient="records")).replace(",", ",\n")

def _ontologies_to_unconnected(ont1, ont2):
    if ont1.strip() == "":
        return ont2
    if ont2.strip() == "":
        return ont1
    try:
        ont1 = json.loads(ont1)
    except:
        ont1 = json.loads(fix_format(ont1))
    try:
        ont2 = json.loads(ont2)
    except:
        ont2 = json.loads(fix_format(ont2))
    for triplet in ont2:
        if triplet not in ont1:
            ont1.append(triplet)
    df = pd.DataFrame(ont1)
    G = nx.Graph()
    for x in df.iloc:
        G.add_edge(x["node_1"], x["node_2"], label=x["edge"])
    dis = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    result = []
    disconnected = []
    for x in dis:
        if x.number_of_nodes() <= 1:
            continue
        res = []
        for u, v, data in x.edges(data=True):
            res.append((u, data["label"], v))
        disconnected.append(res)
    for x in disconnected:
        result.append(_triplets_to_json(x))
    return result

def _combine_ontologies(ont1, ont2, sums, model_id=0):
    """
    Combines two ontologies into a single unified ontology.

    Args:
        ont1: The first ontology.
        ont2: The second ontology.
        sums: The context chunks and their corresponding ontologies.

    Returns:
        The unified ontology in JSON format.

    Raises:
        None.
    """
    disconnected = "\n\n".join(_ontologies_to_unconnected(ont1, ont2))
    prompt = f"""Here's a prompt that takes a series of unconnected ontology graphs and their corresponding context chunks, and generates a single unified ontology in JSON format that combines the individual ontologies with additional linking triplets:
Given a series of unconnected ontology graphs extracted from their respective context chunks, your task is to analyze the contexts and identify potential relationships that could link these isolated ontologies to form a single, cohesive knowledge graph. The goal is to create a unified ontology where all the individual ontologies are connected through meaningful semantic relationships.
Please provide the context chunks and their corresponding ontologies in the following format:
Context chunk:""" + sums + """Ontologys:""" + disconnected + """Please follow these steps to create the unified ontology:""" + open("../prompts/CombineOntologies").read()

    response = str(LLM.generate_chat_response("", prompt, model_id=model_id))
    response = response[response.find("["):response.find("]")+1].lower()
    return response

def _one_switch(ont):
    try:
        ont = json.loads(ont)
    except:
        ont = json.loads(fix_format(ont))
    df = pd.DataFrame(ont)
    G = nx.Graph()
    for x in df.iloc:
        G.add_edge(x["node_1"], x["node_2"], label=x["edge"])
    dis = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    result = []
    disconnected = []
    for x in dis:
        if x.number_of_nodes() <= 1:
            continue
        res = []
        for u, v, data in x.edges(data=True):
            res.append((u, data["label"], v))
        disconnected.append(res)
    for x in disconnected:
        result.append(_triplets_to_json(x))
    return result

def _fix_ontology(ont, context,model_id=0):
    disconnected = "\n\n".join(_one_switch(ont))
    prompt = f"""Here's a prompt that takes a series of unconnected ontology graphs and their corresponding context chunks, and generates a single unified ontology in JSON format that combines the individual ontologies with additional linking triplets:
Given a series of unconnected ontology graphs extracted from their respective context chunks, your task is to analyze the contexts and identify potential relationships that could link these isolated ontologies to form a single, cohesive knowledge graph. The goal is to create a unified ontology where all the individual ontologies are connected through meaningful semantic relationships.
Please provide the context chunks and their corresponding ontologies in the following format:
Context chunk:""" + context + """Ontologys:""" + disconnected + """Please follow these steps to create the unified ontology:""" + open("../prompts/fixOntology").read()
    response = str(LLM.generate_chat_response("", prompt,model_id=model_id))
    response = response[response.find("["):response.find("]")+1].lower()
    return response
