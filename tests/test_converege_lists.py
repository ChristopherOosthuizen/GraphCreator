import pytest

from deepeval import assert_test
import pytest
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
import os 
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
from ..src import GraphCreation as gc



def read_file(file):
    return open(os.path.join(current_dir,"list_converge/"+file)).read()

def test_case():
    list_one = read_file("test1").split("\n\n")
    summaries= read_file("summary_one1").split("\n\n")
    list_one,sums = gc._converge_lists(list_one, summaries= summaries,repeats=1)
    assert len(list_one) == 1