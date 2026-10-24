import sys
import unittest
from pathlib import Path

DIAGNOSTICS = Path(__file__).resolve().parents[1] / "scripts" / "diagnostics"
sys.path.insert(0, str(DIAGNOSTICS))

from sft_sql_parameter_supervision_audit import (  # noqa: E402
    action_exact,
    argument_class,
    classify_error_followup,
    classify_sql_mismatch,
    runtime_branch_strata,
)


class SqlMismatchClassificationTests(unittest.TestCase):
    def test_formatting_only_is_not_identifier_drift(self):
        gold = {
            "kind": "sql",
            "arguments": {"query": "SELECT a,b FROM t"},
        }
        predicted = {
            "kind": "sql",
            "arguments": {"query": "SELECT a, b FROM t"},
        }
        result = classify_sql_mismatch(gold, predicted)
        self.assertEqual(result["primary_category"], "formatting_only")

    def test_identifier_and_filter_drift_are_observable(self):
        gold = {
            "kind": "sql",
            "arguments": {
                "query": "SELECT c.name FROM customers c WHERE c.ordered_at>='2021-01-01'"
            },
        }
        predicted = {
            "kind": "sql",
            "arguments": {
                "query": "SELECT c.customer_name FROM customers c WHERE YEAR(c.order_date)=2021"
            },
        }
        result = classify_sql_mismatch(gold, predicted)
        self.assertEqual(result["primary_category"], "identifier_mismatch")
        self.assertIn("filter_date_dialect_mismatch", result["categories"])

    def test_non_sql_prediction_is_not_called_a_sql_semantic_error(self):
        result = classify_sql_mismatch(
            {"kind": "sql", "arguments": {"query": "SELECT 1"}},
            {"kind": "schema", "arguments": {}},
        )
        self.assertEqual(result["primary_category"], "unclassified")
        self.assertIn("action_kind_mismatch", result["tags"])

    def test_invalid_sql_serialization_is_separate(self):
        result = classify_sql_mismatch(
            {"kind": "sql", "arguments": {"query": "SELECT 1"}},
            None,
            '{"kind":"sql","arguments":{"query":"SELECT\n1"}}',
        )
        self.assertEqual(result["primary_category"], "syntax_or_serialization_error")


class ErrorTransitionTests(unittest.TestCase):
    def test_error_followup_categories(self):
        error = {"action": {"kind": "sql", "arguments": {"query": "SELECT 1"}}}
        same = {"action": {"kind": "sql", "arguments": {"query": "SELECT 1"}}}
        modified = {"action": {"kind": "sql", "arguments": {"query": "SELECT 2"}}}
        self.assertEqual(
            classify_error_followup(error, same, "tool_budget_exhausted"), "retry_same_sql"
        )
        self.assertEqual(
            classify_error_followup(error, modified, "tool_budget_exhausted"), "retry_modified_sql"
        )
        self.assertEqual(
            classify_error_followup(error, {"action": {"kind": "schema", "arguments": {}}}, None),
            "switch_to_schema",
        )
        self.assertEqual(
            classify_error_followup(error, None, "invalid_action_arguments"),
            "invalid_action_or_parser_failure",
        )
        self.assertEqual(
            classify_error_followup(error, None, "tool_budget_exhausted"),
            "terminate_without_next_action",
        )


class ActionBranchAggregationTests(unittest.TestCase):
    def test_argument_classes_and_exact_match(self):
        self.assertEqual(
            argument_class({"kind": "dictionary", "arguments": {"term": "revenue"}}), "term:revenue"
        )
        self.assertEqual(argument_class({"kind": "dictionary", "arguments": {}}), "empty_object")
        self.assertTrue(
            action_exact({"kind": "schema", "arguments": {}}, {"kind": "schema", "arguments": {}})
        )

    def test_runtime_branch_strata_keeps_schema_and_dictionary_separate(self):
        tasks = [
            {
                "trajectory_source": "reference_sql_teacher",
                "template_id": "category",
                "schema_dictionary_before_first_sql_covered": False,
                "actions": [
                    {"kind": "plan", "arguments": {}},
                    {"kind": "sql", "arguments": {}},
                ],
            },
            {
                "trajectory_source": "deterministic_fixture_expert",
                "template_id": "top",
                "schema_dictionary_before_first_sql_covered": True,
                "actions": [
                    {"kind": "plan", "arguments": {}},
                    {"kind": "schema", "arguments": {}},
                    {"kind": "dictionary", "arguments": {}},
                    {"kind": "sql", "arguments": {}},
                ],
            },
        ]
        strata = runtime_branch_strata(tasks, "trajectory_source")
        self.assertEqual(strata["reference_sql_teacher"]["schema_action_count"], 0)
        self.assertEqual(strata["deterministic_fixture_expert"]["schema_action_count"], 1)
        self.assertEqual(strata["deterministic_fixture_expert"]["dictionary_action_count"], 1)
        self.assertEqual(
            strata["deterministic_fixture_expert"][
                "schema_dictionary_before_first_sql_coverage_count"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
