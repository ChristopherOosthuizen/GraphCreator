from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import networkx as nx
import GraphCreation as gc
import textformatting as tx
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
import pandas as pd
import LLMFunctions as lm


def chunks_to_questions(chunks):
    result = []
    for chunk in chunks:
        result.append(lm.generate_chat_response("",chunk+" "+open("../prompts/questionGeneration").read()))
    return result

tokenizer_nli = None
model_nli = None
def follow_premise(answer, chunk):
    if tokenizer_nli is None:
        tokenizer_nli = AutoTokenizer.from_pretrained("potsawee/deberta-v3-large-mnli")
        model_nli = AutoModelForSequenceClassification.from_pretrained("potsawee/deberta-v3-large-mnli")
    inputs = tokenizer_nli.batch_encode_plus(
    batch_text_or_text_pairs=[(answer, chunk)],
    add_special_tokens=True, return_tensors="pt",
    )
    logits = model_nli(**inputs).logits 
    probs = torch.softmax(logits, dim=-1)[0]
    return probs

def llm_as_judge(response1, response2):
    system_prompt = open("../prompts/judgesys").read()
    user_prompt = f"response1: {response1} response2: {response2}"+open("../prompts/judgestandard").read()
    return 1 if "[[A]]" in lm.generate_chat_response(system_prompt, user_prompt) else 0

def llm_benchmark(graph, chunks):
    questions = chunks_to_questions(chunks)
    print(questions)
    results = {"Judges_over_base": [], "Follows_over_base": [], "Controdicts_over_base":[]}
    for i in range(len(questions)):
        question = questions[i]
        base_line = lm.generate_chat_response("", question)
        graph_res = lm.graphquestions(graph, question)
        probs_graph = follow_premise(graph_res,chunks[i])
        probs_base = follow_premise(base_line,chunks[i])
        results["Judges_over_base"].append(llm_as_judge(base_line,graph_res))
        results["Follows_over_base"].append((probs_graph[0].item()-probs_base[0].item()))
        results["Controdicts_over_base"].append(-(probs_graph[1].item()-probs_base[1].item()))
    return results

def networkx_statistics(graph):
    number_of_triangles = sum(nx.triangles(graph).values()) / 3
    number_of_unconnected_graphs = len([x for x in nx.connected_components(graph)])
    connected_graphs = [x for x in nx.connected_components(graph)]
    average_shortest_path = 0
    for x in connected_graphs:
        average_shortest_path += nx.average_shortest_path_length(graph.subgraph(x))
    result = {"bridge_edges": [len(list(nx.bridges(graph))), "Minimize", "Compress",1], #Number of edges that can be removed to disconnect the graph/ Minimize
               "articulation_points": [len(list(nx.articulation_points(graph))), "Minimize", "Compress",1], # Number of nodes that can be removed to connect the graph/ Minimize
               "average_degree": [sum(dict(graph.degree()).values())/graph.number_of_nodes(),"Maximize","Compress",2], # Average number of edges connected to a node/ Maximize
                "efficency": [nx.global_efficiency(graph), "Maximize", "Standard",2], # Inverse of the average shortest path length/ Maximize
                "average_betweenness": [sum(nx.betweenness_centrality(graph).values())/graph.number_of_nodes(),"Maximize", "Standard",2], # Average number of shortest paths that pass through a node/ Maximize
                "average_reaching": [nx.global_reaching_centrality(graph), "Maximize","Standard",1], # The average number of nodes that can be reached from a node/ Maximize
                "number_of_unconneced_graphs": [number_of_unconnected_graphs,"Minimize","Compress",1], # Number of unconnected graphs/ Minimize
                "number_of_triangles": [number_of_triangles, "Maximize","Compress",0], # Number of triangles in the graph/ Maximize
                "number_of_nodes": [graph.number_of_nodes(),"Maximize","Compress",3], # Number of nodes in the graph/ Maximize
                "number_of_edges": [graph.number_of_edges(),"Maximize", "Compress",3], # Number of edges in the graph/ Maximize
                "average_clustering": [nx.average_clustering(graph),"Maximize", "Compress",1], # The clustering coefficient of the graph/ Maximize
                "average_shortest_path": [average_shortest_path/len(connected_graphs),"Minimize","Compress",2] # The average shortest path length of the graph/ Minimize
            }
    return result
def benchmark(graph, chunks):
    llms = llm_benchmark(graph, chunks)
    average_judges = [sum(llms["Judges_over_base"])/len(llms["Judges_over_base"]),"Maximize","Standard",1]# 1 if the graph is better than the base line, 0 if the base line is better/ Maximize
    average_follows = [sum(llms["Follows_over_base"])/len(llms["Follows_over_base"]),"Maximize", "Standard",0] # The difference in the probability of the graph following the base line/ Maximize
    average_contradicts = [sum(llms["Controdicts_over_base"])/len(llms["Controdicts_over_base"]),"Maximize","Standard",0] # The difference in the probability of the graph contradicting the base line/ Maximize
    return {"average_judges": average_judges, "average_follows": average_follows, "average_contradicts": average_contradicts, **networkx_statistics(graph)}

import math
def benchmark_to_score(benchmark):
    result = 0
    priorities = [.5,.3,.15,.05]
    prority_sums = [[]]*len(priorities)
    for key in benchmark:
        num = benchmark[key][0]
        if benchmark[key][2] == "Compress" and num != 0:
            num = 1/math.sqrt(num)
        if benchmark[key][1] == "Minimize":
            num = 1-num
        prority_sums[benchmark[key][3]].append(num)
    for i in range(len(prority_sums)):
        prority_sums[i] = sum(prority_sums[i])/len(prority_sums[i])
    for i in range(len(priorities)):
        result += priorities[i]*prority_sums[i]
    return result

def score(graph, chunks):
    result = benchmark(graph, chunks)
    score = benchmark_to_score(result)
    for key in result:
        result[key] = result[key][0]
    return {"score":score, **result}

def benchmark_params(text, args_list, output="./output/"):
    if not os.path.exists(output):
        os.makedirs(output)
    results = pd.DataFrame()
    for x in range(len(args_list)):
        chunks, graph = gc.create_KG_from_text(text, **args_list[x],output_file=(output+str(x)+"/"))
        main = {**score(graph, chunks),**args_list[x]}
        results._append(main, ignore_index=True)
    return pd.DataFrame(results)

def benchmark_params_url(url, args_list, output_file="./output/"):
    return benchmark_params(tx.url_to_md(url), args_list, output_file)