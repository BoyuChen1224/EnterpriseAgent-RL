"""Deterministic synthetic business data. No oracle tables are stored in the DB."""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path

from enterprise_agent_rl.common import Contract, PublicTask

TABLES = ("regions", "customers", "products", "orders", "order_items")
DICTIONARY = {
    "revenue": "收入仅计 status='paid' 的订单，sum(quantity * unit_price_cents)，单位为分。",
    "orders": (
        "订单：customer_id 关联 customers.id，ordered_at 为 YYYY-MM-DD；paid/cancelled/pending。"
    ),
    "order_items": "订单明细：order_id/product_id 为外键；成交单价 unit_price_cents 是历史快照。",
    "customers": "客户：region_id 关联 regions.id；无订单客户也保留。",
    "products": "产品：category 为 Hardware/Software/Services；目录价格不等于成交价格。",
    "regions": "区域：North/South/East/West；按客户所属区域归属收入。",
}


class BenchmarkTask(Contract):
    public: PublicTask
    split: str
    reference_sql: str
    ordered: bool = False
    min_sql_calls: int = 1
    required_probe_sql: str | None = None


def initialize_database(path: Path, seed: int = 17) -> Path:
    """Create once; refuse to overwrite existing data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement or concurrent initialization.
    with path.open("xb"):
        pass
    rng = random.Random(seed)
    with sqlite3.connect(path) as db:
        db.executescript("""
        PRAGMA foreign_keys=ON;
        CREATE TABLE regions(id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
        CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT NOT NULL,
          region_id INTEGER NOT NULL REFERENCES regions(id));
        CREATE TABLE products(id INTEGER PRIMARY KEY, name TEXT NOT NULL,
          category TEXT NOT NULL, list_price_cents INTEGER NOT NULL CHECK(list_price_cents>0));
        CREATE TABLE orders(id INTEGER PRIMARY KEY,
          customer_id INTEGER NOT NULL REFERENCES customers(id),
          ordered_at TEXT NOT NULL,
          status TEXT NOT NULL CHECK(status IN ('paid','cancelled','pending')));
        CREATE TABLE order_items(order_id INTEGER NOT NULL REFERENCES orders(id),
          product_id INTEGER NOT NULL REFERENCES products(id),
          quantity INTEGER NOT NULL CHECK(quantity>0),
          unit_price_cents INTEGER NOT NULL CHECK(unit_price_cents>0),
          PRIMARY KEY(order_id,product_id));
        CREATE INDEX orders_customer ON orders(customer_id);
        CREATE INDEX orders_date_status ON orders(ordered_at,status);
        """)
        db.executemany(
            "INSERT INTO regions VALUES (?,?)", enumerate(["North", "South", "East", "West"], 1)
        )
        db.executemany(
            "INSERT INTO customers VALUES (?,?,?)",
            [(i, f"Customer-{i:02}", (i - 1) % 4 + 1) for i in range(1, 17)],
        )
        products = [
            (1, "Laptop", "Hardware", 120000),
            (2, "Monitor", "Hardware", 30000),
            (3, "Analytics", "Software", 20000),
            (4, "CRM", "Software", 15000),
            (5, "Consulting", "Services", 50000),
            (6, "Support", "Services", 10000),
        ]
        db.executemany("INSERT INTO products VALUES (?,?,?,?)", products)
        for i in range(1, 97):
            year = 2024 if i <= 48 else 2025
            month = (i - 1) % 12 + 1
            status = "cancelled" if i % 7 == 0 else "pending" if i % 11 == 0 else "paid"
            db.execute(
                "INSERT INTO orders VALUES (?,?,?,?)",
                (i, rng.randint(1, 12), f"{year}-{month:02}-{rng.randint(1, 28):02}", status),
            )
            for p in rng.sample(products, rng.randint(1, 3)):
                db.execute(
                    "INSERT INTO order_items VALUES (?,?,?,?)",
                    (i, p[0], rng.randint(1, 5), p[3] - rng.randint(0, 9) * 100),
                )
    return path


def benchmark_tasks() -> list[BenchmarkTask]:
    tasks = []
    for year, split in [(2024, "train"), (2025, "test")]:
        tasks.append(
            BenchmarkTask(
                public=PublicTask(
                    task_id=f"region-{year}",
                    prompt=(
                        f"统计 {year} 年各区域已支付订单的收入（分），"
                        "返回 region 和 revenue_cents，按 region 排序。"
                    ),
                ),
                split=split,
                ordered=True,
                reference_sql=f"""SELECT r.name AS region,
          SUM(i.quantity*i.unit_price_cents) AS revenue_cents FROM regions r
          JOIN customers c ON c.region_id=r.id JOIN orders o ON o.customer_id=c.id
          JOIN order_items i ON i.order_id=o.id WHERE o.status='paid'
          AND o.ordered_at>='{year}-01-01' AND o.ordered_at<'{year + 1}-01-01'
          GROUP BY r.name ORDER BY region""",
            )
        )
        tasks.append(
            BenchmarkTask(
                public=PublicTask(
                    task_id=f"top-{year}",
                    prompt=(
                        f"先查询 {year} 年已支付订单数量，再找出该年消费最高的前 3 名客户，"
                        "返回 customer 和 revenue_cents，按收入降序、客户名升序。"
                    ),
                ),
                split=split,
                ordered=True,
                min_sql_calls=2,
                required_probe_sql=(
                    f"SELECT COUNT(*) AS paid_orders FROM orders WHERE status='paid' "
                    f"AND ordered_at>='{year}-01-01' "
                    f"AND ordered_at<'{year + 1}-01-01'"
                ),
                reference_sql=f"""SELECT c.name AS customer,
          SUM(i.quantity*i.unit_price_cents) AS revenue_cents FROM customers c
          JOIN orders o ON o.customer_id=c.id JOIN order_items i ON i.order_id=o.id
          WHERE o.status='paid' AND o.ordered_at>='{year}-01-01'
          AND o.ordered_at<'{year + 1}-01-01' GROUP BY c.id
          ORDER BY revenue_cents DESC,customer ASC LIMIT 3""",
            )
        )
        tasks.append(
            BenchmarkTask(
                public=PublicTask(
                    task_id=f"inactive-{year}",
                    prompt=(
                        f"列出 {year} 年没有已支付订单的客户，返回 customer，"
                        "按客户名排序（包括从未下单者）。"
                    ),
                ),
                split=split,
                ordered=True,
                reference_sql=f"""SELECT c.name AS customer FROM customers c
          WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id=c.id AND o.status='paid'
          AND o.ordered_at>='{year}-01-01' AND o.ordered_at<'{year + 1}-01-01')
          ORDER BY customer""",
            )
        )
        for category, cat_split in [
            ("Hardware", split),
            ("Software", "validation" if year == 2024 else split),
        ]:
            tasks.append(
                BenchmarkTask(
                    public=PublicTask(
                        task_id=f"category-{category}-{year}",
                        prompt=(
                            f"统计 {year} 年 {category} 类别已支付订单的收入（分），"
                            "返回 category 和 revenue_cents。"
                        ),
                    ),
                    split=cat_split,
                    reference_sql=f"""SELECT p.category AS category,
              SUM(i.quantity*i.unit_price_cents) AS revenue_cents FROM products p
              JOIN order_items i ON i.product_id=p.id JOIN orders o ON o.id=i.order_id
              WHERE o.status='paid' AND p.category='{category}' AND o.ordered_at>='{year}-01-01'
              AND o.ordered_at<'{year + 1}-01-01' GROUP BY p.category""",
                )
            )
    return tasks
