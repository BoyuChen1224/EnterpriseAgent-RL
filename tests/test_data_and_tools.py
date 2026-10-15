import hashlib
import json
import sqlite3
import time

import pytest

from enterprise_agent_rl.common import Settings
from enterprise_agent_rl.data import benchmark_tasks, initialize_database
from enterprise_agent_rl.tools import SQLTool
from enterprise_agent_rl.trajectories import database_fingerprint


def test_data_reproducible_and_protected(database, tmp_path):
    other = initialize_database(tmp_path / "second.sqlite3")
    assert database_fingerprint(database) == database_fingerprint(other)
    before = database.read_bytes()
    with pytest.raises(FileExistsError):
        initialize_database(database)
    assert database.read_bytes() == before
    with sqlite3.connect(database) as db:
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
        assert db.execute("SELECT count(*) FROM orders").fetchone()[0] == 96
        assert db.execute("SELECT count(*) FROM customers").fetchone()[0] == 16
    tasks = benchmark_tasks()
    assert len({t.public.task_id for t in tasks}) == len(tasks)
    assert len({t.public.prompt for t in tasks}) == len(tasks)
    assert {t.split for t in tasks} == {"train", "validation", "test"}
    assert all("reference_sql" not in t.public.model_dump() for t in tasks)


def test_schema_dictionary_query_and_audit(database, tmp_path, settings):
    audit = tmp_path / "audit.jsonl"
    tool = SQLTool(database, settings, audit)
    assert tool.schema()["orders"]["foreign_keys"][0]["table"] == "customers"
    assert "paid" in tool.dictionary("revenue")["revenue"]
    with pytest.raises(ValueError):
        tool.dictionary("answers")
    sql = "SELECT count(*) AS n FROM orders"
    result = tool.execute(sql, context={"task_id": "audit-test"})
    assert result == {"ok": True, "columns": ["n"], "rows": [[96]], "truncated": False}
    record = json.loads(audit.read_text())
    assert record["sql_sha256"] == hashlib.sha256(sql.encode()).hexdigest()
    assert record["task_id"] == "audit-test"


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM orders",
        "UPDATE orders SET status='paid' RETURNING id",
        "DROP TABLE orders",
        "CREATE TABLE stolen(x)",
        "INSERT INTO regions VALUES(99,'bad')",
        "PRAGMA user_version=5",
        "PRAGMA table_info(orders)",
        "VACUUM",
        "ATTACH DATABASE ':memory:' AS private",
        "SELECT name FROM sqlite_master",
        "SELECT * FROM sqlite_schema",
        "SELECT load_extension('/tmp/not-a-library')",
        "SELECT readfile('/etc/passwd')",
        "SELECT randomblob(1000000000)",
        "SELECT count(*) FROM orders; DELETE FROM orders",
        "/* looks safe */ DELETE FROM orders",
        "WITH x AS (SELECT 1) DELETE FROM orders",
        "BEGIN",
        "SELECT reference_sql FROM answers",
    ],
)
def test_unsafe_sql_denied_and_database_unchanged(database, settings, query):
    before = database_fingerprint(database)
    result = SQLTool(database, settings).execute(query)
    assert result["ok"] is False, result
    assert database_fingerprint(database) == before


def test_sql_limits_and_timeout(database):
    tool = SQLTool(database, Settings(sql_max_rows=2))
    result = tool.execute("SELECT id FROM orders ORDER BY id")
    assert result["rows"] == [[1], [2]] and result["truncated"]
    assert not tool.execute("SELECT 1 AS a,2 AS a")["ok"]
    assert not tool.execute(" " * 16001)["ok"]
    tiny = SQLTool(database, Settings(sql_max_bytes=256))
    assert not tiny.execute("SELECT '" + "x" * 1000 + "' AS large")["ok"]
    started = time.monotonic()
    result = SQLTool(database, Settings(sql_timeout_s=0.2)).execute(
        "SELECT sum(a.id*b.id*c.id*d.id*e.id) FROM orders a "
        "CROSS JOIN orders b CROSS JOIN orders c CROSS JOIN orders d CROSS JOIN orders e"
    )
    assert not result["ok"]
    assert result["error"] in ("query_timeout", "interrupted")
    assert time.monotonic() - started < 4


def test_reference_queries_match_independent_python_aggregation(database, settings):
    from collections import defaultdict

    with sqlite3.connect(database) as db:
        customers = {i: (name, region) for i, name, region in db.execute("SELECT * FROM customers")}
        regions = dict(db.execute("SELECT * FROM regions"))
        products = {
            i: category for i, name, category, price in db.execute("SELECT * FROM products")
        }
        orders = {
            i: (customer, date, status)
            for i, customer, date, status in db.execute("SELECT * FROM orders")
        }
        items = db.execute("SELECT * FROM order_items").fetchall()
    tool = SQLTool(database, settings)
    for task in benchmark_tasks():
        year = task.public.task_id.rsplit("-", 1)[1]
        eligible = {
            i: customer
            for i, (customer, date, status) in orders.items()
            if date.startswith(year) and status == "paid"
        }
        region_revenue, customer_revenue, category_revenue = (defaultdict(int) for _ in range(3))
        for order_id, product_id, quantity, price in items:
            if order_id in eligible:
                customer = eligible[order_id]
                amount = quantity * price
                customer_revenue[customers[customer][0]] += amount
                region_revenue[regions[customers[customer][1]]] += amount
                category_revenue[products[product_id]] += amount
        key = task.public.task_id
        if key.startswith("region"):
            expected = [list(row) for row in sorted(region_revenue.items())]
        elif key.startswith("top"):
            expected = [
                list(row)
                for row in sorted(customer_revenue.items(), key=lambda row: (-row[1], row[0]))[:3]
            ]
        elif key.startswith("inactive"):
            expected = [
                [name]
                for name, region in sorted(customers.values())
                if name not in customer_revenue
            ]
        else:
            category = key.split("-")[1]
            expected = [[category, category_revenue[category]]]
        assert tool.execute(task.reference_sql)["rows"] == expected
        if task.required_probe_sql:
            assert tool.execute(task.required_probe_sql)["rows"] == [[len(eligible)]]


def test_schema_has_hard_process_timeout(database, monkeypatch, settings):
    import subprocess

    def stalled(*args, **kwargs):
        assert kwargs["timeout"] == 0.05
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr("enterprise_agent_rl.tools.subprocess.run", stalled)
    with pytest.raises(ValueError, match="schema_query_timeout"):
        SQLTool(database, settings).schema(timeout_s=0.05)
