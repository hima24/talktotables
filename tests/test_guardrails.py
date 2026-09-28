import duckdb
import pytest

from src.db import QueryTimeout
from src.guardrails import UnsafeSQLError, extract_sql, validate

SAFE = [
    "SELECT COUNT(*) FROM orders",
    "select count(*) from orders;",
    "WITH x AS (SELECT * FROM orders) SELECT COUNT(*) FROM x",
    "SELECT 1 FROM orders UNION ALL SELECT 2 FROM payments",
    "SELECT * FROM main.orders LIMIT 3",
    "SELECT o.order_id FROM orders o WHERE EXISTS (SELECT 1 FROM reviews r WHERE r.order_id = o.order_id)",
    "SELECT seller_id FROM order_items QUALIFY ROW_NUMBER() OVER (PARTITION BY seller_id ORDER BY price) = 1",
]

UNSAFE = [
    "DROP TABLE orders",
    "SELECT 1 FROM orders; DROP TABLE orders",
    "INSERT INTO orders VALUES (1)",
    "UPDATE orders SET order_status = 'x'",
    "DELETE FROM orders",
    "CREATE TABLE t AS SELECT 1",
    "SELECT * FROM read_csv('secrets.csv')",
    "SELECT * FROM 'secrets.csv'",
    "SELECT * FROM users",
    "ATTACH 'other.db'",
    "PRAGMA table_info(orders)",
    "COPY orders TO 'out.csv'",
    "SET enable_external_access = true",
    "SELECT * FROM range(10)",
    "SELECT getenv('HOME')",
    "SELECT * FROM query('SELECT 1')",
    "SELECT * FROM other_db.main.orders",
    "INSTALL httpfs",
    "",
]


@pytest.mark.parametrize("sql", SAFE)
def test_safe_queries_pass(sql):
    validate(sql)


@pytest.mark.parametrize("sql", UNSAFE)
def test_unsafe_queries_blocked(sql):
    with pytest.raises(UnsafeSQLError):
        validate(sql)


def test_extract_sql_from_fenced_reply():
    reply = "Here you go:\n```sql\nSELECT 1 FROM orders;\n```\nThanks"
    assert extract_sql(reply) == "SELECT 1 FROM orders"


def test_database_is_read_only_even_without_guardrails(db):
    # Layer 2: the connection itself refuses writes and file access.
    for sql in ["DROP TABLE orders", "COPY orders TO 'x.csv'", "SELECT * FROM read_csv('/etc/hosts')",
                "SET enable_external_access = true"]:
        with pytest.raises(duckdb.Error):
            db.run(sql)
    assert "orders" in db.tables()


def test_row_limit(db):
    small = type(db)(db.path, max_rows=5)
    out = small.run("SELECT order_id FROM orders ORDER BY order_purchase_timestamp DESC")
    assert out.truncated and len(out.df) == 5
    small.close()


def test_timeout(db):
    slow = type(db)(db.path, timeout_s=1)
    with pytest.raises(QueryTimeout):
        slow.run("SELECT COUNT(*) FROM range(100000000000)")
    slow.close()
