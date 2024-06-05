import pytest

from deepeval import assert_test
import pytest
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
import os 
from ..src import GraphCreation as gc
from ..src import BenchMarks as bm
def test_function():
    chunks, g = gc.create_KG_from_url("https://en.wikipedia.org/wiki/Knight_of_the_shire", chunks_precentage_linked=1, eliminate_all_islands=True, num=15, inital_repeats=2, ner=True, ner_type="llm")
    assert bm.networkx_statistics(g) != None

    chunks, g = gc.create_KG_from_url("", chunks_precentage_linked=1, eliminate_all_islands=True, num=15, inital_repeats=2, ner=True, ner_type="llm")
    assert bm.networkx_statistics(g) != None