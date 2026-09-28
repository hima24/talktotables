# Olist e-commerce database (DuckDB)

Brazilian marketplace orders from Sep 2016 to Oct 2018. Most activity is Jan 2017 - Aug 2018; 2016 and Sep-Oct 2018 are very sparse.

## Tables

### orders (one row per order, ~99K)
- order_id VARCHAR (primary key)
- customer_id VARCHAR -> customers.customer_id (a NEW customer_id is created for every order)
- order_status VARCHAR: delivered, shipped, canceled, unavailable, invoiced, processing, created, approved
- order_purchase_timestamp TIMESTAMP (when the customer bought; use this for "when" questions)
- order_approved_at TIMESTAMP
- order_delivered_carrier_date TIMESTAMP (handed to the carrier)
- order_delivered_customer_date TIMESTAMP (arrived at the customer; NULL if not delivered)
- order_estimated_delivery_date TIMESTAMP (promised date, always at 00:00:00)

### order_items (one row per item in an order, ~113K)
- order_id VARCHAR -> orders.order_id
- order_item_id INTEGER (1, 2, 3 ... position inside the order; NOT a quantity)
- product_id VARCHAR -> products.product_id
- seller_id VARCHAR -> sellers.seller_id
- shipping_limit_date TIMESTAMP
- price DOUBLE (price of this one item, in BRL)
- freight_value DOUBLE (shipping charged for this item, in BRL)

### payments (one or more rows per order, ~104K)
- order_id VARCHAR -> orders.order_id
- payment_sequential INTEGER (1, 2 ... when an order is paid with several methods)
- payment_type VARCHAR: credit_card, boleto, voucher, debit_card, not_defined
- payment_installments INTEGER
- payment_value DOUBLE (BRL)

### reviews (usually one row per order, ~99K)
- review_id VARCHAR (not unique on its own)
- order_id VARCHAR -> orders.order_id (a few orders have more than one review)
- review_score INTEGER 1-5
- review_comment_title VARCHAR (mostly NULL, Portuguese)
- review_comment_message VARCHAR (often NULL, Portuguese)
- review_creation_date TIMESTAMP
- review_answer_timestamp TIMESTAMP

### customers (one row per customer_id, i.e. per order, ~99K)
- customer_id VARCHAR (primary key, joins to orders)
- customer_unique_id VARCHAR (the real person; use this to count customers)
- customer_zip_code_prefix VARCHAR (5 characters)
- customer_city VARCHAR (lowercase, no accents: 'sao paulo', 'rio de janeiro')
- customer_state VARCHAR (2-letter code: SP, RJ, MG, RS, PR, ...)

### sellers (~3K)
- seller_id VARCHAR (primary key)
- seller_zip_code_prefix VARCHAR
- seller_city VARCHAR (lowercase, no accents)
- seller_state VARCHAR (2-letter code)

### products (~33K)
- product_id VARCHAR (primary key)
- product_category_name VARCHAR (Portuguese, e.g. 'beleza_saude'; NULL for ~600 products)
- product_name_length INTEGER, product_description_length INTEGER, product_photos_qty INTEGER
- product_weight_g DOUBLE, product_length_cm DOUBLE, product_height_cm DOUBLE, product_width_cm DOUBLE

### category_translation (71 rows)
- product_category_name VARCHAR -> products.product_category_name
- product_category_name_english VARCHAR (e.g. 'health_beauty')
- A couple of Portuguese categories have no translation, so LEFT JOIN it.

### geolocation (~1M rows, MANY rows per zip prefix)
- geolocation_zip_code_prefix VARCHAR, geolocation_lat DOUBLE, geolocation_lng DOUBLE
- geolocation_city VARCHAR, geolocation_state VARCHAR
- Never join it directly to customers or sellers: aggregate to one row per zip prefix first.

## Join paths
- orders.customer_id = customers.customer_id
- orders.order_id = order_items.order_id = payments.order_id = reviews.order_id
- order_items.product_id = products.product_id
- order_items.seller_id = sellers.seller_id
- products.product_category_name = category_translation.product_category_name

## Traps
- Joining orders to order_items, payments or reviews multiplies rows. Count orders with COUNT(DISTINCT order_id).
- Never join order_items and payments together in one aggregate: both have several rows per order, so sums double-count.
- customers has one row per order. Count people with COUNT(DISTINCT customer_unique_id).
