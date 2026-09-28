"""Load the 9 Olist CSVs from Kaggle into one DuckDB file.

Usage (from the project folder):
    python -m src.load_data

Download the dataset first from
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce and unzip the CSVs
into data/raw/.

Column types are declared explicitly instead of auto-detected, so that
zip-code prefixes keep their leading zeros, timestamps become real TIMESTAMPs,
and a byte-order mark in one of the CSV headers can't break a column name.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd

from src import config

# table name -> (csv file, [(column, type), ...] in file order, expected rows)
TABLES: dict[str, tuple[str, list[tuple[str, str]], int]] = {
    "orders": (
        "olist_orders_dataset.csv",
        [
            ("order_id", "VARCHAR"),
            ("customer_id", "VARCHAR"),
            ("order_status", "VARCHAR"),
            ("order_purchase_timestamp", "TIMESTAMP"),
            ("order_approved_at", "TIMESTAMP"),
            ("order_delivered_carrier_date", "TIMESTAMP"),
            ("order_delivered_customer_date", "TIMESTAMP"),
            ("order_estimated_delivery_date", "TIMESTAMP"),
        ],
        99_441,
    ),
    "order_items": (
        "olist_order_items_dataset.csv",
        [
            ("order_id", "VARCHAR"),
            ("order_item_id", "INTEGER"),
            ("product_id", "VARCHAR"),
            ("seller_id", "VARCHAR"),
            ("shipping_limit_date", "TIMESTAMP"),
            ("price", "DOUBLE"),
            ("freight_value", "DOUBLE"),
        ],
        112_650,
    ),
    "payments": (
        "olist_order_payments_dataset.csv",
        [
            ("order_id", "VARCHAR"),
            ("payment_sequential", "INTEGER"),
            ("payment_type", "VARCHAR"),
            ("payment_installments", "INTEGER"),
            ("payment_value", "DOUBLE"),
        ],
        103_886,
    ),
    "reviews": (
        "olist_order_reviews_dataset.csv",
        [
            ("review_id", "VARCHAR"),
            ("order_id", "VARCHAR"),
            ("review_score", "INTEGER"),
            ("review_comment_title", "VARCHAR"),
            ("review_comment_message", "VARCHAR"),
            ("review_creation_date", "TIMESTAMP"),
            ("review_answer_timestamp", "TIMESTAMP"),
        ],
        99_224,
    ),
    "customers": (
        "olist_customers_dataset.csv",
        [
            ("customer_id", "VARCHAR"),
            ("customer_unique_id", "VARCHAR"),
            ("customer_zip_code_prefix", "VARCHAR"),
            ("customer_city", "VARCHAR"),
            ("customer_state", "VARCHAR"),
        ],
        99_441,
    ),
    "sellers": (
        "olist_sellers_dataset.csv",
        [
            ("seller_id", "VARCHAR"),
            ("seller_zip_code_prefix", "VARCHAR"),
            ("seller_city", "VARCHAR"),
            ("seller_state", "VARCHAR"),
        ],
        3_095,
    ),
    "products": (
        "olist_products_dataset.csv",
        [
            ("product_id", "VARCHAR"),
            ("product_category_name", "VARCHAR"),
            ("product_name_length", "INTEGER"),  # "lenght" typo fixed
            ("product_description_length", "INTEGER"),  # "lenght" typo fixed
            ("product_photos_qty", "INTEGER"),
            ("product_weight_g", "DOUBLE"),
            ("product_length_cm", "DOUBLE"),
            ("product_height_cm", "DOUBLE"),
            ("product_width_cm", "DOUBLE"),
        ],
        32_951,
    ),
    "category_translation": (
        "product_category_name_translation.csv",
        [
            ("product_category_name", "VARCHAR"),
            ("product_category_name_english", "VARCHAR"),
        ],
        71,
    ),
    "geolocation": (
        "olist_geolocation_dataset.csv",
        [
            ("geolocation_zip_code_prefix", "VARCHAR"),
            ("geolocation_lat", "DOUBLE"),
            ("geolocation_lng", "DOUBLE"),
            ("geolocation_city", "VARCHAR"),
            ("geolocation_state", "VARCHAR"),
        ],
        1_000_163,
    ),
}

ZIP_COLUMNS = {
    "customer_zip_code_prefix",
    "seller_zip_code_prefix",
    "geolocation_zip_code_prefix",
}


def _load_with_duckdb(con, table, path: Path, cols):
    spec = "{" + ", ".join(f"'{c}': '{t}'" for c, t in cols) + "}"
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT * FROM read_csv(
            '{path.as_posix()}',
            header = true,
            columns = {spec},
            quote = '"',
            escape = '"',
            timestampformat = '%Y-%m-%d %H:%M:%S'
        )
        """
    )


def _load_with_pandas(con, table, path: Path, cols):
    """Fallback if DuckDB's CSV reader rejects a file (e.g. odd quoting)."""
    names = [c for c, _ in cols]
    df = pd.read_csv(path, header=0, names=names, dtype=str, keep_default_na=True)
    con.register("tmp_df", df)
    casts = []
    for c, t in cols:
        if t == "VARCHAR":
            casts.append(f'"{c}"')
        else:
            casts.append(f'TRY_CAST("{c}" AS {t}) AS "{c}"')
    con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT {', '.join(casts)} FROM tmp_df")
    con.unregister("tmp_df")


def _pad_zips(con, table, cols):
    for c, _ in cols:
        if c in ZIP_COLUMNS:
            con.execute(f"UPDATE {table} SET {c} = lpad({c}, 5, '0') WHERE {c} IS NOT NULL")


def load(raw_dir: Path = config.RAW_DIR, db_path: Path = config.DB_PATH, check_counts: bool = True) -> dict:
    raw_dir, db_path = Path(raw_dir), Path(db_path)
    missing = [f for f, _, _ in TABLES.values() if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing CSVs in {raw_dir}: {missing}\n"
            "Download https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce and unzip into data/raw/"
        )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    counts = {}
    for table, (fname, cols, expected) in TABLES.items():
        path = raw_dir / fname
        try:
            _load_with_duckdb(con, table, path, cols)
            how = "duckdb"
        except duckdb.Error as e:
            print(f"  ! DuckDB reader failed on {fname} ({str(e).splitlines()[0]}), using pandas")
            _load_with_pandas(con, table, path, cols)
            how = "pandas"
        _pad_zips(con, table, cols)
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        counts[table] = n
        flag = ""
        if check_counts and n != expected:
            flag = f"   <-- expected {expected:,}, check this file"
        print(f"  {table:<22}{n:>10,} rows  ({how}){flag}")

    # Empty strings in text columns become NULL so "IS NULL" checks behave.
    for table, (_, cols, _) in TABLES.items():
        for c, t in cols:
            if t == "VARCHAR":
                con.execute(f"UPDATE {table} SET {c} = NULL WHERE trim({c}) = ''")

    con.execute("CHECKPOINT")
    con.close()
    print(f"Saved {db_path}")
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=config.RAW_DIR)
    ap.add_argument("--db", default=config.DB_PATH)
    ap.add_argument("--no-count-check", action="store_true", help="skip the expected-row-count check (for sample data)")
    args = ap.parse_args(argv)
    try:
        load(args.raw_dir, args.db, check_counts=not args.no_count_check)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
