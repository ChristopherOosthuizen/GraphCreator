from deepeval import assert_test
import pytest
from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase
import os 
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
prompts_dir = os.path.join(current_dir, '..', 'src','prompts')
from ..src import GraphCreation as gc

def read_file(file):
    return open(os.path.join(current_dir,"extract_tests/"+file)).read()

def test_case():
    input_one = read_file("test1")
    input_two = read_file("test2")

    metric = SummarizationMetric(0.8)
    summary_test = LLMTestCase(
        input=open(os.path.join(prompts_dir,"summaryPrompt")).read()+"context1: "+input_one+" context2: "+input_two,
        retrieval_context=[input_one, input_two],
        actual_output=gc.new_summary_prompt(input_one, input_two),
    )
    assert_test(summary_test, [metric])
