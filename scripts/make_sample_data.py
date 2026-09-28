"""Generate a SMALL, SYNTHETIC dataset with the exact Olist file names and columns.

This is only for automated tests and a no-download smoke test. The numbers it
produces mean nothing. Real results always come from the Kaggle dataset.

    python scripts/make_sample_data.py --out data/sample_raw --orders 600
"""
from __future__ import annotations

import argparse
import csv
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

STATES = ["SP", "SP", "SP", "RJ", "RJ", "MG", "MG", "RS", "PR", "BA", "SC", "GO", "PE", "CE", "DF",
          "ES", "MT", "MS", "PA", "PB", "RN", "AL", "PI", "MA", "SE", "TO", "RO", "AM", "AC", "AP", "RR"]
CITIES = {"SP": "sao paulo", "RJ": "rio de janeiro", "MG": "belo horizonte", "RS": "porto alegre",
          "PR": "curitiba", "BA": "salvador"}
CATEGORIES = [
    ("cama_mesa_banho", "bed_bath_table"),
    ("beleza_saude", "health_beauty"),
    ("esporte_lazer", "sports_leisure"),
    ("moveis_decoracao", "furniture_decor"),
    ("informatica_acessorios", "computers_accessories"),
    ("utilidades_domesticas", "housewares"),
    ("relogios_presentes", "watches_gifts"),
    ("telefonia", "telephony"),
    ("pc_gamer", None),  # exists in products but has no English translation, like the real data
]
STATUSES = ["delivered"] * 17 + ["shipped", "canceled", "unavailable", "invoiced", "processing"]
PAY_TYPES = ["credit_card"] * 7 + ["boleto"] * 2 + ["voucher", "debit_card"]


def uid(rng):
    return uuid.UUID(int=rng.getrandbits(128)).hex


def ts(d):
    return d.strftime("%Y-%m-%d %H:%M:%S") if d else ""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sample_raw")
    ap.add_argument("--orders", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sellers = []
    for _ in range(40):
        st = rng.choice(STATES[:10])
        sellers.append([uid(rng), f"{rng.randint(1000, 99999):05d}", CITIES.get(st, "cidade"), st])
    products = []
    for _ in range(120):
        cat = rng.choice(CATEGORIES + [(None, None)])[0]
        products.append([uid(rng), cat or "", rng.randint(20, 60), rng.randint(100, 2000), rng.randint(1, 5),
                         rng.randint(100, 20000), rng.randint(10, 80), rng.randint(2, 50), rng.randint(10, 60)])

    people = [(uid(rng), rng.choice(STATES)) for _ in range(int(args.orders * 0.9))]
    customers, orders, items, payments, reviews = [], [], [], [], []
    start = datetime(2016, 10, 1)
    for i in range(args.orders):
        unique_id, st = rng.choice(people)
        cust_id = uid(rng)  # Olist gives every order a new customer_id
        customers.append([cust_id, unique_id, f"{rng.randint(1000, 99999):05d}", CITIES.get(st, "cidade"), st])
        oid = uid(rng)
        bought = start + timedelta(days=rng.randint(0, 700), seconds=rng.randint(0, 86399))
        status = rng.choice(STATUSES)
        approved = bought + timedelta(hours=rng.randint(0, 48)) if status != "created" else None
        carrier = approved + timedelta(days=rng.randint(1, 5)) if status in ("delivered", "shipped") else None
        est = datetime(bought.year, bought.month, bought.day) + timedelta(days=rng.randint(15, 35))
        delivered = None
        if status == "delivered" and rng.random() > 0.02:
            delivered = bought + timedelta(days=rng.randint(3, 40), hours=rng.randint(0, 23))
        orders.append([oid, cust_id, status, ts(bought), ts(approved), ts(carrier), ts(delivered), ts(est)])

        n_items = rng.choices([1, 2, 3, 6], weights=[80, 12, 6, 2])[0]
        prod = rng.choice(products)
        seller = rng.choice(sellers)
        total = 0.0
        for k in range(1, n_items + 1):
            if rng.random() < 0.3:
                prod, seller = rng.choice(products), rng.choice(sellers)
            price = round(rng.uniform(5, 900), 2)
            freight = round(rng.uniform(5, 60), 2)
            total += price + freight
            items.append([oid, k, prod[0], seller[0], ts(bought + timedelta(days=3)), price, freight])

        if rng.random() < 0.08:
            split = round(total * 0.3, 2)
            payments.append([oid, 1, "voucher", 1, split])
            payments.append([oid, 2, "credit_card", rng.randint(1, 10), round(total - split, 2)])
        else:
            pt = rng.choice(PAY_TYPES)
            payments.append([oid, 1, pt, rng.randint(1, 12) if pt == "credit_card" else 1, round(total, 2)])

        if rng.random() < 0.97:
            late = delivered and delivered.date() > est.date()
            score = rng.choices([1, 2, 3, 4, 5], weights=[40, 10, 15, 20, 15] if late else [5, 3, 8, 24, 60])[0]
            msg = rng.choice(["", "", "Muito bom, recomendo", "Produto chegou atrasado,\nnao gostei", 'Otimo "produto"'])
            created = (delivered or bought + timedelta(days=20)) + timedelta(days=1)
            reviews.append([uid(rng), oid, score, "", msg, ts(created), ts(created + timedelta(days=2))])
            if rng.random() < 0.01:  # a few orders have two reviews, like the real data
                reviews.append([uid(rng), oid, rng.randint(1, 5), "", "", ts(created), ts(created)])

    geo = []
    for st in STATES:
        for _ in range(3):  # duplicates per zip prefix, like the real table
            geo.append([f"{rng.randint(1000, 99999):05d}", round(rng.uniform(-30, -3), 6),
                        round(rng.uniform(-60, -35), 6), CITIES.get(st, "cidade"), st])

    def write(name, header, rows, bom=False):
        with open(out / name, "w", newline="", encoding="utf-8-sig" if bom else "utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    write("olist_orders_dataset.csv", ["order_id", "customer_id", "order_status", "order_purchase_timestamp",
          "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date",
          "order_estimated_delivery_date"], orders)
    write("olist_order_items_dataset.csv", ["order_id", "order_item_id", "product_id", "seller_id",
          "shipping_limit_date", "price", "freight_value"], items)
    write("olist_order_payments_dataset.csv", ["order_id", "payment_sequential", "payment_type",
          "payment_installments", "payment_value"], payments)
    write("olist_order_reviews_dataset.csv", ["review_id", "order_id", "review_score", "review_comment_title",
          "review_comment_message", "review_creation_date", "review_answer_timestamp"], reviews)
    write("olist_customers_dataset.csv", ["customer_id", "customer_unique_id", "customer_zip_code_prefix",
          "customer_city", "customer_state"], customers)
    write("olist_sellers_dataset.csv", ["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"], sellers)
    write("olist_products_dataset.csv", ["product_id", "product_category_name", "product_name_lenght",
          "product_description_lenght", "product_photos_qty", "product_weight_g", "product_length_cm",
          "product_height_cm", "product_width_cm"], products)
    write("product_category_name_translation.csv", ["product_category_name", "product_category_name_english"],
          [[pt, en] for pt, en in CATEGORIES if en], bom=True)
    write("olist_geolocation_dataset.csv", ["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng",
          "geolocation_city", "geolocation_state"], geo)
    print(f"Wrote synthetic sample to {out} ({len(orders)} orders, {len(items)} items)")


if __name__ == "__main__":
    main()
