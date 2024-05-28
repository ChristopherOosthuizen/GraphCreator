from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
import LLMFunctions as lm

tokenizer_question = T5Tokenizer.from_pretrained("iarfmoose/t5-base-question-generator")
model_question = T5ForConditionalGeneration.from_pretrained("iarfmoose/t5-base-question-generator")
def chunks_to_questions(chunks):
    result = []
    for chunk in chunks:
        input_ids = tokenizer_question(chunk, return_tensors="pt").input_ids
        outputs = model_question.generate(input_ids)
        result.append(tokenizer_question.decode(outputs[0]))
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
    results = {"Judges_over_base": [], "Follows_over_base": [], "Controdicts_over_base":[]}
    for i,question in questions:
        base_line = lm.generate_chat_response("", question)
        graph_res = lm.graphquestions(graph, question)
        probs_graph = follow_premise(graph_res,chunks[i])
        probs_base = follow_premise(base_line,chunks[i])
        results["Judges_over_base"].append(llm_as_judge(base_line,graph_res))
        results["Follows_over_base"].append(probs_graph[1]-probs_base[1])
        results["Controdicts_over_base"].append(-probs_graph[0]-probs_base[0])
    return results