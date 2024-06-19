import json
import pandas as pd
import networkx as nx

import os 
from . import LLMFunctions as LLM
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
prompts_dir = os.path.join(current_dir, '..', 'prompts')
import re
def _triplets_to_json(triplets):
    data = []
    for triplet in triplets:
        data.append(triplet[0]+","+triplet[1]+","+triplet[2])
    return "\n".join(data)

def _ontologies_to_unconnected(ont1, ont2):
    G = nx.Graph()
    for x in ont1.split("\n"):
        objects = x.split(",")
        if len(objects) < 3:
            print(objects)
            continue
        G.add_edge(objects[0], objects[2], label=objects[1])
    for x in ont2.split("\n"):
        objects = x.split(",")
        if len(objects) < 3:
            print(objects)
            continue
        G.add_edge(objects[0], objects[2], label=objects[1])
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
    return response

def _one_switch(ont):
    G = nx.Graph()
    for x in ont.split("\n"):
        objects = x.split(",")
        if len(objects) < 3:
            print(objects)
            continue
        G.add_edge(objects[0], objects[2], label=objects[1])
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
    return response
