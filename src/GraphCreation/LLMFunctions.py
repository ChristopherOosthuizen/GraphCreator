from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from llama_index.core import StorageContext
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import Settings
from llama_index.core.chat_engine import ContextChatEngine
from openai import OpenAI
from transformers import pipeline
import torch
import os
import numpy as np
from llama_index.llms.huggingface import HuggingFaceLLM
pipelines = []
gpus = os.environ['KG_GPUS'].split(",")
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
if "HF_HOME" in os.environ:
    for x in gpus:
        pipelines.append(pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map=int(x),
            ))
def pipeline_select():
    """
    Selects a pipeline for the LLM.

    Returns:
        pipeline: The selected pipeline.
    """
    memory_usage = [torch.cuda.memory_reserved(int(x))-torch.cuda.memory_allocated(int(x)) for x in gpus]
    return pipelines[np.argmin(memory_usage)]
def generate_chat_response(system_prompt, user_prompt):
    """
    Generates a chat response using OpenAI's GPT-4o model.

    Args:
        system_prompt (str): The system prompt for the chat.
        user_prompt (str): The user prompt for the chat.

    Returns:
        str: The generated chat response.
    """
    messages = messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    if not "HF_HOME" in os.environ:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )
        return response.choices[0].message.content
    pipeline = pipeline_select()
    prompter = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompter,
        max_new_tokens=3000,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.01,
        top_p=0.5,
        repetition_penalty=1.1,
    )
    return outputs[0]["generated_text"][len(prompter):]

def graphquestions(graph, prompt):
    """
    Function to ask questions about a graph.

    Args:
        graph (Graph): The graph to ask questions about.
        prompt (str): The question prompt.

    Returns:
        str: The response to the question.
    """
    if "HF_HOME" in os.environ:
        pipeline = pipeline_select()
        Settings.llm = HuggingFaceLLM(model_name=model_id, model=pipeline.model,tokenizer=pipeline.tokenizer)
    graph_store = SimpleGraphStore()
    for node_1, node_2, data in graph.edges(data=True):
        graph_store.upsert_triplet(node_1, data['label'], node_2)
    
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    graph_rag_retriever = KnowledgeGraphRAGRetriever(
        storage_context=storage_context,
        verbose=False,
    )
    chat_engine = ContextChatEngine.from_defaults(
        retriever=graph_rag_retriever,
        verbose=False,
        max_entities=30,
        graph_traversal_depth=10,
        max_knowledge_sequence=15,
    )
    
    response = chat_engine.chat(prompt)
    return response.response