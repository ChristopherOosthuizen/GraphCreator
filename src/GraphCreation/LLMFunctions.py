from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from llama_index.core import StorageContext
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import Settings
from llama_index.core.chat_engine import ContextChatEngine
from openai import OpenAI
from transformers import pipeline
import torch
import numpy as np
from llama_index.llms.huggingface import HuggingFaceLLM

from llama_index.core import SummaryIndex
from llama_index.readers.web import SimpleWebPageReader
from llama_index.llms.ollama import Ollama
import google.generativeai as genai
llm = None 
model = None
pipelines = []
gpus = []
model_id = ""
import os
def set_model(model_name: str):
    global model
    if model_name.startswith("gpt"):
        assert 'OPENAI_API_KEY' in os.environ, "The OpenAI API key must be set in the environment variables. This can be done by os.environ['OPENAI_API_KEY'] = 'your_key_here'"
        llm = OpenAI()
        model = model_name
    elif model_name.startswith('gemini'):
        assert 'GENAI_API_KEY' in os.environ, "The GenAI API key must be set in the environment variables. This can be done by os.environ['GENAI_API_KEY'] = 'your_key_here'"
        genai.configure(api_key=os.environ['GENAI_API_KEY'])
        llm = genai.GenerativeModel(model_name)
        model = "genai"
    elif model_name.find("/") != -1:
        assert 'HF_HOME' in os.environ, "The Hugging Face API key must be set in the environment variables. This can be done by os.environ['HF_HOME'] = 'your_key_here'"
        llm = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16})
        model = "huggingface"
    else:
        assert model_name.startswith("phi3") or model_name.startswith("llama3") or model_name.startswith("mistral") or model_name.startswith("gemma2"), "The model name must start with 'phi3' or 'llama3' or 'mistral' or 'gemma2'"
        llm = Ollama(model=model_name)
        llm.complete("respond with nothing to this message")
        model = "ollama"
def pick_gpu(index):
    """
    Function to pick a GPU for the pipeline.

    Args:
        index (int): The index of the GPU to pick.

    Returns:
        int: The GPU picked.
    """
    if 'KG_GPUS' not in os.environ:
        return 0
    gpu_length = len(os.environ['KG_GPUS'].split(","))
    return index%gpu_length
index = 0
def generate_chat_response(system_prompt, user_prompt, model_id=0):
    """
    Generates a chat response using OpenAI's GPT-4o model.

    Args:
        system_prompt (str): The system prompt for the chat.
        user_prompt (str): The user prompt for the chat.

    Returns:
        str: The generated chat response.
    """
    assert llm is not None, "The LLM must be set before calling this function. use set_model(model_name)"
    messages = messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    assert model.startswith("gpt") or model == 'genai' or model == 'huggingface' or model=='ollama', "Invalid model name. Must be 'gpt' or 'genai' or 'huggingface' or 'ollama'"
    if model.startswith("gpt"):
        return str(llm.chat.completions.create(messages=messages, model=model)).lower()
    elif model == 'genai':
        return str(llm.generate_content(messages[0]["content"]+" "+ messages[1]["content"])).lower()
    elif model == 'ollama':
        return str(llm.complete(messages[0]["content"]+" "+ messages[1]["content"])).lower()
    return str(pipelines[model_id](messages[0]["content"]+" "+ messages[1]["content"])).lower()

def graphquestions(graph, prompt, pipeline_id=0):
    """
    Function to ask questions about a graph.

    Args:
        graph (Graph): The graph to ask questions about.
        prompt (str): The question prompt.

    Returns:
        str: The response to the question.
    """
    if "HF_HOME" in os.environ:
        global index
        pipeline_id = pick_gpu(index)
        index += 1
        pipeline = pipelines[pipeline_id]
        Settings.llm = HuggingFaceLLM(model_name=model_id, model=pipeline.model,tokenizer=pipeline.tokenizer)
    Settings.llm = llm
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

def doRag(url, question, pipeline_id=0):
    """
    Function to ask questions about a graph.

    Args:
        url (str): The URL of the graph.
        question (str): The question prompt.

    Returns:
        str: The response to the question.
    """
    if "HF_HOME" in os.environ:
        global index
        pipeline_id = pick_gpu(index)
        index += 1
        pipeline = pipelines[pipeline_id]
        Settings.llm = HuggingFaceLLM(model_name=model_id, model=pipeline.model,tokenizer=pipeline.tokenizer)
    Settings.llm = llm
    documents = SimpleWebPageReader(html_to_text=True).load_data([url])
    
    index = SummaryIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    
    response = query_engine.query(question)
    return response.response