from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import networkx as nx
from transformers import set_seed
import pandas as pd
import json
import threading
from . import GraphCreation as gc
from . import LLMFunctions as lm
from . import textformatting as tx
from . import LinkPrediction as lp
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
prompts_dir = os.path.join(current_dir, '..', 'prompts')
def chunks_to_questions(chunks):
    result = []
    for chunk in chunks:
        result.append(lm.generate_chat_response("",chunk+" "+open(os.path.join(prompts_dir,"questionGeneration")).read()))
    return result

tokenizer_nli = AutoTokenizer.from_pretrained("potsawee/deberta-v3-large-mnli")
model_nli = AutoModelForSequenceClassification.from_pretrained("potsawee/deberta-v3-large-mnli")
def follow_premise(answer, chunk):
    inputs = tokenizer_nli.batch_encode_plus(
    batch_text_or_text_pairs=[(answer, chunk)],
    add_special_tokens=True, return_tensors="pt",
    )
    logits = model_nli(**inputs).logits 
    probs = torch.softmax(logits, dim=-1)[0]
    return probs

def llm_as_judge(response1, response2):
    system_prompt = open(os.path.join(prompts_dir,"judgesys")).read()
    user_prompt = f"response1: {response1} response2: {response2}"+open(os.path.join(prompts_dir,"judgestandard")).read()
    return 1 if "[[A]]" in lm.generate_chat_response(system_prompt, user_prompt) else 0

def llm_benchmark(graph, chunks):
    questions = chunks_to_questions(chunks)
    print(questions)
    results = {"Judges_over_base": [], "Follows_over_base": [], "Controdicts_over_base":[], "base_line": [], "graph": []}
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
    result = {"bridge_edges": [len(list(nx.bridges(graph))), "Minimize", "Compress",1], #Number of edges that can be removed to disconnect the graph/ Minimizeß
               "average_degree": [sum(dict(graph.degree()).values())/graph.number_of_nodes(),"Maximize","Compress",2], # Average number of edges connected to a node/ Maximize
                "efficency": [nx.global_efficiency(graph), "Maximize", "Standard",2], # Inverse of the average shortest path length/ Maximize
                "average_betweenness": [sum(nx.betweenness_centrality(graph).values())/graph.number_of_nodes(),"Maximize", "Standard",3], # Average number of shortest paths that pass through a node/ Maximize
                "average_reaching": [nx.global_reaching_centrality(graph), "Maximize","Standard",3], # The average number of nodes that can be reached from a node/ Maximize
                "number_of_unconneced_graphs": [number_of_unconnected_graphs,"Minimize","Compress",1], # Number of unconnected graphs/ Minimize
                "number_of_triangles": [number_of_triangles, "Maximize","Compress",1], # Number of triangles in the graph/ Maximize
                "number_of_nodes": [graph.number_of_nodes(),"Maximize","Compress",4], # Number of nodes in the graph/ Maximize
                "number_of_edges": [graph.number_of_edges(),"Maximize", "Compress",4], # Number of edges in the graph/ Maximize
                "average_clustering": [nx.average_clustering(graph),"Maximize", "Compress",2], # The clustering coefficient of the graph/ Maximize
                "average_shortest_path": [average_shortest_path/len(connected_graphs),"Minimize","Compress",4] # The average shortest path length of the graph/ Minimize
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
    priorities = [.5,.2,.15,.1,.05]
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
    results = []
    for x in range(len(args_list)):
        try:
            chunks, graph = gc.create_KG_from_text(text, **args_list[x],output_file=(output+str(x)+"/"))
            main = {**score(graph, chunks),**args_list[x]}
            results.append(main)
        except:
            results.append({**args_list[x], "score": 0})
    return pd.DataFrame(results)

def benchmark_params_url(url, args_list, output_file="./output/"):
    return benchmark_params(tx.url_to_md(url), args_list, output_file)

def single_DPO(text, list, index, seed):
    set_seed(seed)
    jsons = gc._create_kg(chunks=[text], converge=False, repeats=0, inital_repeats=0, ner=True, ner_type="llm")
    Graph = nx.Graph()
    for x in jsons:
        try:
            x = json.loads(x)
        except:
            x = json.loads(lp.fix_format(x))
        for y in x:
            Graph.add_edge(y["node_1"], y["node_2"], label=y["edge"])
    list[index] = {**score(Graph, [text]), "output": jsons}
def create_DPO(file, output_file="./output/"):
    resulter = []
    chunks = tx.chunk_text(tx.pdf_to_md(file))
    seeds = [0,100,200,300,400]
    for c in range(len(chunks)):
        outputs = [""]*5
        threads = []
        for y in range(5):
            if not os.path.exists(output_file):
                os.makedirs(output_file)
            single_DPO(chunks[c], outputs, y, seeds[y])
        maxs = outputs[0]
        mins = outputs[0]
        for z in outputs:
            if z["score"] > maxs["score"]:
                maxs = z
            if z["score"] < mins["score"]:
                mins = z
        inputs = open(os.path.join(prompts_dir,"TripletCreationSystem")).read()+ f"Context: ```{chunks[c]}``` \n\nOutput: "
        result_output = {"input":inputs ,"taken":maxs['output'], "rejected":mins['output'], "taken_score":maxs["score"], "rejected_score":mins["score"]}
        resulter.append(result_output)
        pd.DataFrame(resulter).to_csv(output_file+"results.csv")
    return pd.DataFrame(resulter)
 
def create_DPO_folder(folder, output_file="./output/"):
    files = os.listdir(folder)
    results = []
    for file in files:
        results.append(create_DPO(folder+"/"+file, output_file+file+"/"))
    resulter = pd.concat(results)
    pd.DataFrame(resulter).to_csv(output_file+"results.csv")
    return resulter
    
def bench_mark_from_dataset(dataframe, source_column, answer_column, question_column,  output_file="./output/",eliminate_all_islands=False, inital_repeats=2, chunks_precentage_linked=0.5, ner=False, ner_type="flair" ):
    result = []
    for x in range(len(dataframe)):
        try:
            url = dataframe[source_column].iloc[x]
            chunks, graph = gc.create_KG_from_url(url, output_file+str(x), eliminate_all_islands=eliminate_all_islands, inital_repeats=inital_repeats, chunks_precentage_linked=chunks_precentage_linked, ner=ner, ner_type=ner_type,llm_formatting=False)
            question = dataframe[question_column].iloc[x]
            answer = dataframe[answer_column].iloc[x]
            base_line = lm.generate_chat_response("", question)
            graph_res = lm.graphquestions(graph, question)
            rag_res = lm.doRag(url, question)
            probs_rag = follow_premise(rag_res,chunks[0])
            probs_graph = follow_premise(graph_res,chunks[0])
            probs_base = follow_premise(base_line,chunks[0])
            main = {"Judges_over_base": llm_as_judge(base_line,graph_res),"Follows_over_rag": (probs_graph[0].item()-probs_rag[0].item()), "Follows_over_base": (probs_graph[0].item()-probs_base[0].item()), "Judges_over_rag": llm_as_judge(rag_res,graph_res), "Controdicts_over_base": -(probs_graph[1].item()-probs_base[1].item()),"Controdicts_over_rag": -(probs_graph[1].item()-probs_rag[1].item()), "base_line": base_line, "graph": graph_res,"rag":rag_res, "source": url, "answer": answer, "question": question}
            result.append(main)
            pd.DataFrame(result).to_csv(output_file+"results.csv")
        except:
            print("Error")
    return result