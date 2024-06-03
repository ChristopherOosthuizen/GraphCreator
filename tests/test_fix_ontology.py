import pytest

from deepeval import assert_test
import pytest
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
import os 
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
from ..src import LinkPrediction as lp

def test_function():
    ont = lp._fix_ontology(open(os.path.join(current_dir,"fix_ontologyTests/testOne")).read(),"")
    assert len(lp._ontologies_to_unconnected(ont,ont)) == 1

    ont = lp._fix_ontology(open(os.path.join(current_dir,"fix_ontologyTests/testTwo")).read(),open(os.path.join(current_dir,"fix_ontologyTests/testTwoChunk")).read())
    assert len(lp._ontologies_to_unconnected(ont,ont)) == 1
    ont = lp._fix_ontology(open(os.path.join(current_dir,"fix_ontologyTests/testThree")).read(),open(os.path.join(current_dir,"fix_ontologyTests/testThreeChunk")).read())
    assert len(lp._ontologies_to_unconnected(ont,ont)) == 1
    ont = lp._fix_ontology(open(os.path.join(current_dir,"fix_ontologyTests/testFour")).read(),open(os.path.join(current_dir,"fix_ontologyTests/testFourChunk")).read())
    assert len(lp._ontologies_to_unconnected(ont,ont)) == 1