import json
import pandas as pd
import networkx as nx

import os 
from . import LLMFunctions as LLM
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
prompts_dir = os.path.join(current_dir, '..', 'prompts')
import re
def strip_json(jsons):
    jsons = re.sub(r'(?m)^ *#.*\n', '', jsons)
    jsons = re.sub(r'(?m)^ *//.*\n', '', jsons)
    jsons.replace("\n\n","\n")
    return jsons
def fix_format(input, error="", model_id=0):
    """
    Fixes the format of the input JSON.

    Args:
        input (str): The input JSON to fix the format of.

    Returns:
        str: The fixed format of the input JSON.
    """
    prompt = f"given this error {error} and json \nOriginal Json: {input}" + open(os.path.join(prompts_dir,"TripletCreationSystem")).read()
    response = str(LLM.generate_chat_response("", prompt, model_id=model_id))
    response = response[response.find("["):response.find("]")+1]
    return response

def _triplets_to_json(triplets):
    df = pd.DataFrame({"node_1": [], "node_2": [], "edge": []})
    for triplet in triplets:
        df = df._append({"node_1": triplet[0], "node_2": triplet[2], "edge": triplet[1]}, ignore_index=True)
    return json.dumps(df.to_dict(orient="records")).replace(",", ",\n")

def _ontologies_to_unconnected(ont1, ont2):
    if not ont1.strip():
        return ont2
    if not ont2.strip():
        return ont1
    try:
        ont1 = json.loads(ont1)
    except Exception as err:
        print(ont1)
        ont1 = json.loads(fix_format(ont1, str(err)))
    try:
        ont2 = json.loads(ont2)
    except Exception as err:
        ont2 = json.loads(fix_format(ont2, str(err)))
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
    chunks = _one_switch(ont1)+_one_switch(ont2)
    if len(chunks) == 1:
        return _ontologies_to_unconnected(ont1, ont2)[0]
    result = "Conext: "+sums+"\n\n"
    for x in range(len(chunks)):
        result += "Chunk "+str(x+1)+":\n"+chunks[x]+"\n\n"
    prompt = result
    response = str(LLM.generate_chat_response(open(os.path.join(prompts_dir,"Fusionsys")).read(), prompt,model_id=model_id))
    response = response[response.find("["):response.find("]")+1].lower()
    response = strip_json(response)
    return response

def _one_switch(ont):
    try:
        ont = json.loads(ont)
    except Exception as err:
        ont = json.loads(fix_format(ont, str(err)))
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
    chunks = _one_switch(ont)
    if len(chunks) == 1:
        return ont
    result = "Conext: "+context+"\n\n"
    for x in range(len(chunks)):
        result += "Chunk "+str(x+1)+":\n"+chunks[x]+"\n\n"
    prompt = result
    response = str(LLM.generate_chat_response(open(os.path.join(prompts_dir,"Fusionsys")).read(), prompt,model_id=model_id))
    response = response[response.find("["):response.find("]")+1].lower()
    response = strip_json(response)
    return response
