from deepeval import assert_test
import pytest
from deepeval.metrics import ContextualPrecisionMetric, AnswerRelevancyMetric, HallucinationMetric
from deepeval.test_case import LLMTestCase
import os 
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
from ..src import GraphCreation as gc
import json
def read_file(file):
    return open(os.path.join(current_dir,"extract_tests/"+file)).read()
    

def test_case():
    input_one = read_file("test1")
    jsons = json.loads(gc.create_knowledge_triplets(input_one,repeats=0, num=10,ner=False))
    assert len(jsons) >= 10

    input_one = read_file("test2")
    jsons = json.loads(gc.create_knowledge_triplets(input_one,repeats=0, num=15,ner=False))
    assert len(jsons) >= 15

    input_one = read_file("test3")
    jsons = json.loads(gc.create_knowledge_triplets(input_one,repeats=0, num=8,ner=False))
    assert len(jsons) >= 8

    input_one = read_file("test4")
    jsons = json.loads(gc.create_knowledge_triplets(input_one,repeats=0, num=20,ner=False))
    assert len(jsons) >= 20

    input_one = read_file("test1")
    jsons = json.loads(gc.create_knowledge_triplets(input_one,repeats=1, num=10,ner=False))
    assert len(jsons) >= 11

    input_one = read_file("test2")
    jsons = json.loads(gc.create_knowledge_triplets(input_one,repeats=2, num=15,ner=False))
    assert len(jsons) >= 17

    input_one = read_file("test3")
    jsons = json.loads(gc.create_knowledge_triplets(input_one,repeats=3, num=8,ner=False))
    assert len(jsons) >= 11

    input_one = read_file("test4")
    jsons = json.loads(gc.create_knowledge_triplets(input_one,repeats=5, num=20,ner=False))
    assert len(jsons) >= 25

    input_one = read_file("test1")
    jsons = json.loads(gc.create_knowledge_triplets(input_one,repeats=0, num=10,ner=True))
    assert len(jsons) >= 10

    input_one = read_file("test2")
    jsons = json.loads(gc.create_knowledge_triplets(input_one,repeats=0, num=15,ner=True))
    assert len(jsons) >= 15

    input_one = read_file("test3")
    jsons = json.loads(gc.create_knowledge_triplets(input_one,repeats=0, num=10,ner=True, ner_type="llm"))
    assert len(jsons) >= 15