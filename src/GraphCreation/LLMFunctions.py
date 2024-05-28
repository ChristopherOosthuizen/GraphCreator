from llama_index.core import set_global_tokenizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from llama_index.core import StorageContext
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import Settings
from llama_index.core.chat_engine import ContextChatEngine
from openai import OpenAI

def generate_chat_response(system_prompt, user_prompt):
    """
    Generates a chat response using OpenAI's GPT-4o model.

    Args:
        system_prompt (str): The system prompt for the chat.
        user_prompt (str): The user prompt for the chat.

    Returns:
        str: The generated chat response.
    """
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return response.choices[0].message.content

def graphquestions(graph, prompt):
    """
    Function to ask questions about a graph.

    Args:
        graph (Graph): The graph to ask questions about.
        prompt (str): The question prompt.

    Returns:
        str: The response to the question.
    """
    graph_store = SimpleGraphStore()
    for node_1, node_2, data in graph.edges(data=True):
        graph_store.upsert_triplet(node_1, data['title'], node_2)
    
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