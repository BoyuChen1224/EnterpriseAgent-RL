"""M2A synthetic benchmark: manifest is verifier-side, never policy input."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from enterprise_agent_rl.common import Settings
from enterprise_agent_rl.data import BenchmarkTask, PublicTask, initialize_database
from enterprise_agent_rl.tools import SQLTool
from enterprise_agent_rl.trajectories import database_fingerprint

VERSION = "m2a-synthetic-v1"
SPLITS = {"train": (2021, 101), "validation": (2023, 201), "test": (2025, 301)}


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def create_instance(path: Path, split: str, instance: int) -> None:
    year, seed = SPLITS[split]
    initialize_database(path, seed + instance)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE orders SET ordered_at = ? || substr(ordered_at,5)", (str(year),))
        db.execute("UPDATE customers SET name = ? || name", (f"{split}-{instance}-",))
        db.execute("UPDATE products SET name = ? || name", (f"{split}-{instance}-",))
        db.commit()
        db.execute("VACUUM")


def task_specs(split: str, instance: int) -> list[tuple[str, str, str, dict]]:
    year, _ = SPLITS[split]
    prefix = f"{split}-{instance}-"
    where = f"o.status='paid' AND o.ordered_at>='{year}-01-01' AND o.ordered_at<'{year + 1}-01-01'"
    join = (
        " FROM orders o JOIN order_items i ON i.order_id=o.id "
        "JOIN customers c ON c.id=o.customer_id JOIN products p ON p.id=i.product_id "
    )
    money = "SUM(i.quantity*i.unit_price_cents)"
    count = f"SELECT COUNT(*) AS paid_orders FROM orders o WHERE {where}"
    specs = [
        (
            "schema",
            "先使用 schema 确认客户表字段，列出客户名称和区域编号，"
            "返回 customer, region_id，按客户名排序。",
            "SELECT name AS customer,region_id FROM customers ORDER BY customer",
            {"required_schema": True},
        ),
        (
            "region",
            f"统计 {year} 年各区域已支付订单收入（分），"
            "返回 region, revenue_cents，按 region 排序。",
            f"SELECT r.name AS region,{money} AS revenue_cents"
            + join
            + f"JOIN regions r ON r.id=c.region_id WHERE {where} GROUP BY r.name ORDER BY region",
            {},
        ),
        (
            "top",
            f"先查询 {year} 年已支付订单数量（列 paid_orders），再列消费最高的前 3 名客户，"
            "返回 customer, revenue_cents，收入降序、客户名升序。",
            f"SELECT c.name AS customer,{money} AS revenue_cents"
            + join
            + f"WHERE {where} GROUP BY c.id ORDER BY revenue_cents DESC,customer LIMIT 3",
            {"min_sql_calls": 2, "required_probe_sql": count},
        ),
        (
            "customer",
            f"统计 {year} 年客户 {prefix}Customer-03 的已支付收入（分）；即使无成交也返回一行。"
            "返回 customer, revenue_cents。",
            "SELECT c.name AS customer,"
            "COALESCE(SUM(i.quantity*i.unit_price_cents),0) AS revenue_cents "
            "FROM customers c LEFT JOIN orders o ON o.customer_id=c.id AND "
            + where
            + " LEFT JOIN order_items i ON i.order_id=o.id "
            f"WHERE c.name='{prefix}Customer-03' GROUP BY c.id",
            {},
        ),
        (
            "product",
            f"统计 {year} 年商品 {prefix}Laptop 的已支付销量和收入（分），"
            "返回 product, units, revenue_cents。",
            f"SELECT p.name AS product,SUM(i.quantity) AS units,{money} AS revenue_cents"
            + join
            + f"WHERE {where} AND p.name='{prefix}Laptop' GROUP BY p.id",
            {},
        ),
        (
            "empty",
            f"列出 {year + 5} 年有已支付订单的客户，返回 customer，按客户名排序。",
            "SELECT DISTINCT c.name AS customer FROM customers c "
            "JOIN orders o ON o.customer_id=c.id "
            f"WHERE o.status='paid' AND o.ordered_at>='{year + 5}-01-01' "
            f"AND o.ordered_at<'{year + 6}-01-01' ORDER BY customer",
            {},
        ),
        (
            "constraints",
            f"统计 {year} 年 North 区域 Hardware 类商品的已支付收入（分），"
            "只返回收入至少 100000 分的客户，列 customer, revenue_cents；收入降序、客户名升序。",
            f"SELECT c.name AS customer,{money} AS revenue_cents"
            + join
            + f"JOIN regions r ON r.id=c.region_id WHERE {where} "
            "AND r.name='North' AND p.category='Hardware' GROUP BY c.id "
            f"HAVING {money}>=100000 ORDER BY revenue_cents DESC,customer",
            {},
        ),
        (
            "recovery",
            "验证历史报表 SQL `SELECT total_amount FROM orders LIMIT 1` 是否仍可执行，"
            "保留该次实际执行的错误反馈，再根据当前 schema 修正报表。"
            f"最终返回 {year} 年已支付订单数，列 paid_orders。",
            count,
            {"required_sql_error": True, "required_schema": True},
        ),
    ]
    # Two template families per split are absent from the other splits.
    if split == "train":
        specs += [
            (
                "category",
                f"统计 {year} 年各类别已支付收入，返回 category, revenue_cents，类别升序。",
                f"SELECT p.category,{money} AS revenue_cents"
                + join
                + f"WHERE {where} GROUP BY p.category ORDER BY category",
                {},
            ),
            (
                "inactive",
                f"列出 {year} 年没有已支付订单的客户（包括从未下单），列 customer，客户名升序。",
                "SELECT c.name AS customer FROM customers c WHERE NOT EXISTS "
                f"(SELECT 1 FROM orders o WHERE o.customer_id=c.id AND {where}) ORDER BY customer",
                {},
            ),
        ]
    elif split == "validation":
        specs += [
            (
                "status",
                f"统计 {year} 年每种状态的订单数，列 status, order_count，状态升序。",
                "SELECT status,COUNT(*) AS order_count FROM orders "
                f"WHERE ordered_at>='{year}-01-01' AND ordered_at<'{year + 1}-01-01' "
                "GROUP BY status ORDER BY status",
                {},
            ),
            (
                "monthly",
                f"统计 {year} 年每月已支付收入，列 month, revenue_cents，月份升序。",
                f"SELECT substr(o.ordered_at,1,7) AS month,{money} AS revenue_cents"
                + join
                + f"WHERE {where} GROUP BY month ORDER BY month",
                {},
            ),
        ]
    else:
        specs += [
            (
                "distinct",
                f"统计 {year} 年各区域有已支付订单的去重客户数，"
                "列 region, customer_count，区域升序。",
                "SELECT r.name AS region,COUNT(DISTINCT c.id) AS customer_count FROM regions r "
                "JOIN customers c ON c.region_id=r.id JOIN orders o ON o.customer_id=c.id "
                f"WHERE {where} GROUP BY r.name ORDER BY region",
                {},
            ),
            (
                "basket",
                f"统计 {year} 年每个已支付订单的商品总件数，只保留至少 5 件的订单，"
                "返回 order_id, units，按 order_id 升序。",
                "SELECT o.id AS order_id,SUM(i.quantity) AS units FROM orders o "
                f"JOIN order_items i ON i.order_id=o.id WHERE {where} "
                "GROUP BY o.id HAVING SUM(i.quantity)>=5 ORDER BY order_id",
                {},
            ),
        ]
    return specs


def python_oracle(database: Path, template: str, split: str, instance: int) -> list[list]:
    """Independent relational calculation, without reference SQL or SQL aggregation."""
    with sqlite3.connect(database) as db:
        db.row_factory = sqlite3.Row
        tables = {
            table: [dict(r) for r in db.execute(f"SELECT * FROM {table}")]
            for table in ("customers", "products", "regions", "orders", "order_items")
        }
    customers = {r["id"]: r for r in tables["customers"]}
    products = {r["id"]: r for r in tables["products"]}
    regions = {r["id"]: r["name"] for r in tables["regions"]}
    orders = {r["id"]: r for r in tables["orders"]}
    year, _ = SPLITS[split]
    paid = {
        k: v
        for k, v in orders.items()
        if v["status"] == "paid" and v["ordered_at"].startswith(str(year))
    }
    values = defaultdict(int)
    units = defaultdict(int)
    for item in tables["order_items"]:
        if item["order_id"] not in paid:
            continue
        order = paid[item["order_id"]]
        customer, product = customers[order["customer_id"]], products[item["product_id"]]
        region = regions[customer["region_id"]]
        key = {
            "region": region,
            "top": customer["name"],
            "customer": customer["name"],
            "product": product["name"],
            "category": product["category"],
            "constraints": customer["name"],
            "monthly": order["ordered_at"][:7],
            "basket": order["id"],
        }.get(template)
        if template == "constraints" and (region != "North" or product["category"] != "Hardware"):
            continue
        values[key] += item["quantity"] * item["unit_price_cents"]
        units[key] += item["quantity"]
    if template == "schema":
        return sorted([[r["name"], r["region_id"]] for r in customers.values()])
    if template == "empty":
        return []
    if template == "recovery":
        return [[len(paid)]]
    if template == "top":
        return sorted([[k, v] for k, v in values.items()], key=lambda r: (-r[1], r[0]))[:3]
    if template == "constraints":
        return sorted(
            [[k, v] for k, v in values.items() if v >= 100000], key=lambda r: (-r[1], r[0])
        )
    if template == "customer":
        name = f"{split}-{instance}-Customer-03"
        return [[name, values[name]]]
    if template == "product":
        name = f"{split}-{instance}-Laptop"
        return [[name, units[name], values[name]]] if name in values else []
    if template == "inactive":
        active = {o["customer_id"] for o in paid.values()}
        return sorted([[c["name"]] for k, c in customers.items() if k not in active])
    if template == "status":
        counts = Counter(
            o["status"] for o in orders.values() if o["ordered_at"].startswith(str(year))
        )
        return sorted([[k, v] for k, v in counts.items()])
    if template == "distinct":
        grouped = defaultdict(set)
        for o in paid.values():
            c = customers[o["customer_id"]]
            grouped[regions[c["region_id"]]].add(c["id"])
        return sorted([[k, len(v)] for k, v in grouped.items()])
    if template == "basket":
        return sorted([[k, v] for k, v in units.items() if v >= 5])
    return sorted([[k, v] for k, v in values.items()])


def build_manifest(directory: Path) -> dict:
    """Create reproducible DBs, validate every label against a Python oracle."""
    directory.mkdir(parents=True, exist_ok=True)
    records, databases = [], []
    for split in SPLITS:
        for instance in range(2):
            db_id = f"{split}-{instance}"
            path = directory / f"{db_id}.sqlite3"
            create_instance(path, split, instance)
            db_hash = database_fingerprint(path)
            databases.append(
                {
                    "id": db_id,
                    "split": split,
                    "sha256": db_hash,
                    "seed": SPLITS[split][1] + instance,
                    "year": SPLITS[split][0],
                    "database_version": VERSION,
                }
            )
            for template, prompt, query, constraints in task_specs(split, instance):
                task = BenchmarkTask(
                    public=PublicTask(task_id=f"{db_id}-{template}", prompt=prompt),
                    split=split,
                    reference_sql=query,
                    ordered=True,
                    **constraints,
                )
                result = SQLTool(path, Settings(sql_timeout_s=10)).execute(query)
                expected = python_oracle(path, template, split, instance)
                if not result["ok"] or result["truncated"] or result["rows"] != expected:
                    raise RuntimeError(f"reference_oracle_mismatch:{task.public.task_id}:{result}")
                record = {
                    "task": task.model_dump(),
                    "database_id": db_id,
                    "database_version": VERSION,
                    "database_sha256": db_hash,
                    "source": "project-authored synthetic task; no external benchmark",
                    "template": template,
                    "label_validation": "SQL vs independent Python",
                    "expected_result_sha256": canonical_hash(
                        {"columns": result["columns"], "rows": result["rows"]}
                    ),
                }
                record["data_fingerprint"] = canonical_hash(record)
                records.append(record)
    manifest = {
        "version": VERSION,
        "databases": databases,
        "records": records,
        "split_counts": dict(Counter(r["task"]["split"] for r in records)),
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    return manifest


def load_frozen_manifest(path: Path, database_root: Path) -> dict:
    manifest = json.loads(path.read_text())
    declared = manifest.pop("manifest_sha256")
    if canonical_hash(manifest) != declared:
        raise ValueError("manifest_hash_mismatch")
    lock = path.with_suffix(".sha256").read_text().strip()
    if declared != lock:
        raise ValueError("frozen_manifest_changed")
    for record in manifest["records"]:
        fields = {k: v for k, v in record.items() if k != "data_fingerprint"}
        if canonical_hash(fields) != record["data_fingerprint"]:
            raise ValueError("task_fingerprint_mismatch")
    for entry in manifest["databases"]:
        if database_fingerprint(database_root / (entry["id"] + ".sqlite3")) != entry["sha256"]:
            raise ValueError("benchmark_database_changed")
    manifest["manifest_sha256"] = declared
    return manifest
