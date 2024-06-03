from deepeval import assert_test
import pytest
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
import os 
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
from ..src import textformatting as tx


def test_case():
    input_one = open("src/prompts/formatting").read()+" "+open(os.path.join(current_dir,"textformattingTests/testOne")).read()
    presion_metric = ContextualPrecisionMetric()
    knight_case = LLMTestCase(
        input=input_one,
        expected_output=open(os.path.join(current_dir,"textformattingTests/testOneOut")).read(),
        retrieval_context=[open(os.path.join(current_dir,"textformattingTests/testOneOut")).read()],
        actual_output=tx.format_text(input_one, ""),
    )
    assert_test(knight_case, [presion_metric])
    input_one = open("src/prompts/formatting").read()+" "+open(os.path.join(current_dir,"textformattingTests/testTwo")).read()
    presion_metric = ContextualPrecisionMetric()
    docs_case = LLMTestCase(
        input=input_one,
        expected_output=open(os.path.join(current_dir,"textformattingTests/testTwoOut")).read(),
        retrieval_context=[open(os.path.join(current_dir,"textformattingTests/testTwoOut")).read()],
        actual_output=tx.format_text(input_one, ""),
    )
    assert_test(docs_case, [presion_metric])
    input_one = open("src/prompts/formatting").read()+" "+open(os.path.join(current_dir,"textformattingTests/testThree")).read()
    presion_metric = ContextualPrecisionMetric()
    modern_case = LLMTestCase(
        input=input_one,
        expected_output=open(os.path.join(current_dir,"textformattingTests/testThreeOut")).read(),
        retrieval_context=[open(os.path.join(current_dir,"textformattingTests/testThreeOut")).read()],
        actual_output=tx.format_text(input_one, ""),
    )
    assert_test(modern_case, [presion_metric])
    