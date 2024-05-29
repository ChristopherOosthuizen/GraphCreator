from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import networkx as nx
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
import LLMFunctions as lm


def chunks_to_questions(chunks):
    result = []
    for chunk in chunks:
        result.append(lm.generate_chat_response("",chunk+" "+open("../prompts/questionGeneration").read()))
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
    result = {"bridge_edges": len(list(nx.bridges(graph))), #Number of edges that can be removed to disconnect the graph/ Minimize
               "articulation_points": len(list(nx.articulation_points(graph))), # Number of nodes that can be removed to connect the graph/ Minimize
               "average_degree": sum(dict(graph.degree()).values())/graph.number_of_nodes(), # Average number of edges connected to a node/ Maximize
                "efficency": nx.global_efficiency(graph), # Inverse of the average shortest path length/ Maximize
                "average_betweenness": sum(nx.betweenness_centrality(graph).values())/graph.number_of_nodes(), # Average number of shortest paths that pass through a node/ Maximize
                "average_reaching": nx.global_reaching_centrality(graph), # The average number of nodes that can be reached from a node/ Maximize
                "connectivity": nx.node_connectivity(graph), # The minimum number of nodes that need to be removed to disconnect the graph/ Maximize
                "number_of_unconneced_graphs": number_of_unconnected_graphs, # Number of unconnected graphs/ Minimize
                "number_of_triangles": number_of_triangles, # Number of triangles in the graph/ Maximize
                "number_of_nodes": graph.number_of_nodes(), # Number of nodes in the graph/ Maximize
                "number_of_edges": graph.number_of_edges(), # Number of edges in the graph/ Maximize
                "average_clustering": nx.average_clustering(graph), # The clustering coefficient of the graph/ Maximize
                "average_shortest_path": average_shortest_path/len(connected_graphs) # The average shortest path length of the graph/ Minimize
            }
    return result
def benchmark(graph, chunks):
    llms = llm_benchmark(graph, chunks)
    average_judges = sum(llms["Judges_over_base"])/len(llms["Judges_over_base"])# 1 if the graph is better than the base line, 0 if the base line is better/ Maximize
    average_follows = sum(llms["Follows_over_base"])/len(llms["Follows_over_base"]) # The difference in the probability of the graph following the base line/ Maximize
    average_contradicts = sum(llms["Controdicts_over_base"])/len(llms["Controdicts_over_base"]) # The difference in the probability of the graph contradicting the base line/ Maximize
    return {"average_judges": average_judges, "average_follows": average_follows, "average_contradicts": average_contradicts, **networkx_statistics(graph)}


