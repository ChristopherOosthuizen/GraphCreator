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
    ms= {"bridge_edges": len(list(nx.bridges(graph))), "articulation_points": len(list(nx.articulation_points(graph)))}
    other = {"average_degree": sum(dict(graph.degree()).values())/graph.number_of_nodes(), "efficency": nx.global_efficiency(graph)}
    function_metrics = {"average_betweenness": sum(nx.betweenness_centrality(graph).values())/graph.number_of_nodes(), "average_reaching": nx.global_reaching_centrality(graph), "connectivity": nx.node_connectivity(graph)}
    basic_metrics = {"number_of_unconneced_graphs": number_of_unconnected_graphs, "number_of_triangles": number_of_triangles, "number_of_nodes": graph.number_of_nodes(), "number_of_edges": graph.number_of_edges(),}
    right = { "average_clustering": nx.average_clustering(graph), "average_shortest_path": average_shortest_path/len(connected_graphs)}
    return { **basic_metrics,**right, **function_metrics, **other, **ms}
def benchmark(graph, chunks):
    llms = llm_benchmark(graph, chunks)
    average_judges = sum(llms["Judges_over_base"])/len(llms["Judges_over_base"])
    average_follows = sum(llms["Follows_over_base"])/len(llms["Follows_over_base"])
    average_contradicts = sum(llms["Controdicts_over_base"])/len(llms["Controdicts_over_base"])
    return {"average_judges": average_judges, "average_follows": average_follows, "average_contradicts": average_contradicts, **networkx_statistics(graph)}


