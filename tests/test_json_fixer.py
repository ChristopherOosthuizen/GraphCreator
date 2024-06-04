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
    presion_metric = ContextualPrecisionMetric()
    ont = lp.fix_format(open(os.path.join(current_dir,"textformattingTests/fixformatOne")).read())
    knight_case = LLMTestCase(
        input=open(os.path.join(current_dir,"textformattingTests/fixformatOne")).read(),
        expected_output=open(os.path.join(current_dir,"textformattingTests/fixformateOneOut")).read(),
        retrieval_context=[open(os.path.join(current_dir,"textformattingTests/fixformateOneOut")).read()],
        actual_output=ont,
        )
    assert_test(knight_case, [presion_metric])
    
    ont = lp.fix_format(open(os.path.join(current_dir,"textformattingTests/fixformatTwo")).read())
    knight_case = LLMTestCase(
        input=open(os.path.join(current_dir,"textformattingTests/fixformatTwo")).read(),
        expected_output=open(os.path.join(current_dir,"textformattingTests/fixformatTwoOut")).read(),
        retrieval_context=[open(os.path.join(current_dir,"textformattingTests/fixformatTwoOut")).read()],
        actual_output=ont,
        )
    assert_test(knight_case, [presion_metric])

    ont = lp.fix_format(open(os.path.join(current_dir,"textformattingTests/fixformatThree")).read())
    knight_case = LLMTestCase(
        input=open(os.path.join(current_dir,"textformattingTests/fixformatThree")).read(),
        expected_output=open(os.path.join(current_dir,"textformattingTests/fixformatThreeOut")).read(),
        retrieval_context=[open(os.path.join(current_dir,"textformattingTests/fixformatThreeOut")).read()],
        actual_output=ont,
        )
    assert_test(knight_case, [presion_metric])

    ont = lp.fix_format(open(os.path.join(current_dir,"textformattingTests/fixformatFour")).read())
    knight_case = LLMTestCase(
        input=open(os.path.join(current_dir,"textformattingTests/fixformatFour")).read(),
        expected_output=open(os.path.join(current_dir,"textformattingTests/fixformatFourOut")).read(),
        retrieval_context=[open(os.path.join(current_dir,"textformattingTests/fixformatFourOut")).read()],
        actual_output=ont,
        )
    assert_test(knight_case, [presion_metric])

    ont = lp.fix_format(open(os.path.join(current_dir,"textformattingTests/fixformatFive")).read())
    knight_case = LLMTestCase(
        input=open(os.path.join(current_dir,"textformattingTests/fixformatFive")).read(),
        expected_output=open(os.path.join(current_dir,"textformattingTests/fixformatFiveOut")).read(),
        retrieval_context=[open(os.path.join(current_dir,"textformattingTests/fixformatFiveOut")).read()],
        actual_output=ont,
        )
    assert_test(knight_case, [presion_metric])

    ont = lp.fix_format(open(os.path.join(current_dir,"textformattingTests/fixformatSix")).read())
    knight_case = LLMTestCase(
        input=open(os.path.join(current_dir,"textformattingTests/fixformatSix")).read(),
        expected_output=open(os.path.join(current_dir,"textformattingTests/fixformatSixOut")).read(),
        retrieval_context=[open(os.path.join(current_dir,"textformattingTests/fixformatSixOut")).read()],
        actual_output=ont,
        )
    assert_test(knight_case, [presion_metric])

    ont = lp.fix_format(open(os.path.join(current_dir,"textformattingTests/fixformatSeven")).read())
    knight_case = LLMTestCase(
        input=open(os.path.join(current_dir,"textformattingTests/fixformatSeven")).read(),
        expected_output=open(os.path.join(current_dir,"textformattingTests/fixformatSevenOut")).read(),
        retrieval_context=["Since the context is not none it should output a empty list like []"],
        actual_output=ont,
        )
    assert_test(knight_case, [presion_metric])