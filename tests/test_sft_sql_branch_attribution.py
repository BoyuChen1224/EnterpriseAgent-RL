import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "diagnostics"))

import sft_sql_branch_attribution as audit  # noqa: E402


class SftSqlBranchAttributionTest(unittest.TestCase):
    def test_branch_class(self):
        self.assertEqual(audit.branch_class(True, True), "schema_and_dictionary")
        self.assertEqual(audit.branch_class(True, False), "schema_only")
        self.assertEqual(audit.branch_class(False, True), "dictionary_only")
        self.assertEqual(audit.branch_class(False, False), "neither")

    def test_saved_error_text_classification(self):
        self.assertEqual(
            audit.classify_runtime_error_category(
                {"tool_result_ok": False, "error": "no such column: i.unit_price_cents"}
            ),
            "identifier_mismatch",
        )
        self.assertEqual(
            audit.classify_runtime_error_category(
                {"tool_result_ok": False, "error": "syntax error near SELECT"}
            ),
            "syntax_or_serialization_error",
        )
        self.assertEqual(
            audit.classify_runtime_error_category(
                {
                    "tool_result_ok": False,
                    "error": "tool_budget_exhausted",
                    "error_is_budget_or_terminal_sentinel": True,
                }
            ),
            "budget_or_terminal_sentinel",
        )

    def test_runtime_strata_strict_recovery_and_repeat(self):
        task = {
            "group": "baseline",
            "group_label": "A",
            "task_id": "train-0-top",
            "database_id": "train-0",
            "template_id": "top",
            "trajectory_source": "deterministic_fixture_expert",
            "actions": [{"step_index": 0, "kind": "schema"}],
            "parser_or_action_terminal": False,
            "budget_terminal": False,
            "terminal_reason": "check",
        }
        raw_events = [
            {
                "group": "baseline",
                "group_label": "A",
                "task_id": "train-0-top",
                "database_id": "train-0",
                "template_id": "top",
                "trajectory_source": "deterministic_fixture_expert",
                "sql_event_index": 0,
                "step_index": 1,
                "query": "SELECT bad_col FROM orders",
                "query_sha256": "q0",
                "tool_result_ok": False,
                "error": "no such column: bad_col",
                "error_identifier": "bad_col",
                "error_is_budget_or_terminal_sentinel": False,
                "next_action": "sql",
                "gold_comparison": {},
                "source_files": {"result": "result.json"},
            },
            {
                "group": "baseline",
                "group_label": "A",
                "task_id": "train-0-top",
                "database_id": "train-0",
                "template_id": "top",
                "trajectory_source": "deterministic_fixture_expert",
                "sql_event_index": 1,
                "step_index": 2,
                "query": "SELECT bad_col FROM orders",
                "query_sha256": "q0",
                "tool_result_ok": False,
                "error": "no such column: bad_col",
                "error_identifier": "bad_col",
                "error_is_budget_or_terminal_sentinel": False,
                "next_action": "sql",
                "gold_comparison": {},
                "source_files": {"result": "result.json"},
            },
            {
                "group": "baseline",
                "group_label": "A",
                "task_id": "train-0-top",
                "database_id": "train-0",
                "template_id": "top",
                "trajectory_source": "deterministic_fixture_expert",
                "sql_event_index": 2,
                "step_index": 3,
                "query": "SELECT good_col FROM orders",
                "query_sha256": "q1",
                "tool_result_ok": True,
                "error": None,
                "error_identifier": None,
                "error_is_budget_or_terminal_sentinel": False,
                "next_action": None,
                "gold_comparison": {"exact_gold_sequence_indices": [0]},
                "source_files": {"result": "result.json"},
            },
        ]
        task["sql_events"] = raw_events
        events = audit.build_runtime_sql_events([task])
        self.assertEqual(
            [event["preceding_accepted_branch_class"] for event in events],
            ["schema_only", "schema_only", "schema_only"],
        )
        self.assertTrue(events[0]["exact_query_repeat"])
        self.assertTrue(events[0]["later_successful_sql_after_error"])
        self.assertTrue(events[1]["modified_query_after_error"])
        stats = audit.aggregate_runtime_strata(events, [task])["schema_only"]
        self.assertEqual(stats["execution_error_event_count"], 2)
        self.assertEqual(stats["exact_query_repeat_count"], 1)
        self.assertEqual(stats["successful_sql_after_prior_sql_error_event_count"], 2)
        self.assertEqual(stats["failed_task_count"], 1)

    def test_teacher_gold_prefix_is_not_runtime_acceptance(self):
        task = {
            "task_id": "train-0-product",
            "database_id": "train-0",
            "template_id": "product",
            "trajectory_source": "reference_sql_teacher",
            "source_file": "dataset.jsonl",
            "source_trajectory_path": "trajectory.json",
            "actions": [
                {
                    "sequence_index": 0,
                    "assistant_message_index": 3,
                    "kind": "plan",
                    "preceding_action_kinds": [],
                },
                {
                    "sequence_index": 1,
                    "assistant_message_index": 5,
                    "kind": "dictionary",
                    "arguments": {"term": "revenue"},
                    "preceding_action_kinds": ["plan"],
                },
                {
                    "sequence_index": 2,
                    "assistant_message_index": 7,
                    "kind": "sql",
                    "arguments": {"query": "SELECT revenue FROM products"},
                    "preceding_action_kinds": ["plan", "dictionary"],
                    "sql_visibility": {"status": "unconfirmed_no_visible_schema"},
                },
            ],
        }
        record = {
            "prediction_index": 2,
            "target_id": "target",
            "task_id": "train-0-product",
            "database_id": "train-0",
            "template_id": "product",
            "trajectory_source": "reference_sql_teacher",
            "assistant_message_index": 7,
            "gold_kind": "sql",
            "predicted_kind": "sql",
            "gold_action": {"kind": "sql", "arguments": {"query": "SELECT revenue FROM products"}},
            "predicted_action": {"kind": "sql", "arguments": {"query": "SELECT bad FROM products"}},
            "gold_query": "SELECT revenue FROM products",
            "predicted_query": "SELECT bad FROM products",
            "parameter_exact": False,
            "exact_match": False,
            "sql_mismatch": {"primary_category": "identifier_mismatch"},
            "source_file": "predictions.jsonl",
        }
        event = audit.build_teacher_sql_events([task], [record])[0]
        self.assertIsNone(event["preceding_schema_accepted"])
        self.assertFalse(event["gold_prefix_schema_visible"])
        self.assertTrue(event["gold_prefix_dictionary_visible"])
        self.assertEqual(event["gold_prefix_branch_class"], "dictionary_only")
        self.assertEqual(event["error_category"], "not_executed_teacher_forced")


if __name__ == "__main__":
    unittest.main()
