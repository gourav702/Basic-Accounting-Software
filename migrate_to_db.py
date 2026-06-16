"""
One-time migration: copy your local CSV data into Postgres.

Run this ONCE after creating your Neon database, with DATABASE_URL pointing at it:

    DATABASE_URL='postgresql://...neon.../db?sslmode=require' python3 migrate_to_db.py

It reads items.csv and invoices.csv from this folder and loads them into the
Postgres tables (creating the tables first). Safe to re-run: it replaces the
table contents with whatever is currently in the CSV files.
"""
import csv
import os
import sys

import server  # imports in Postgres mode because DATABASE_URL is set

if not server.USE_PG:
    sys.exit("DATABASE_URL is not set — nothing to migrate to. "
             "Set it to your Neon connection string and re-run.")

server.init_db()

# --- items.csv -> items table ---
items = []
with open(server.ITEMS_CSV, newline='') as f:
    for r in csv.DictReader(f):
        items.append({'id': int(r['id']), 'name': r['name'],
                      'qty': int(r['qty']), 'price': float(r['price'])})
server.save_items(items)

# --- invoices.csv -> invoices table (handles the 9-column schema) ---
invoices = []
with open(server.INVOICES_CSV, newline='') as f:
    for r in csv.DictReader(f):
        invoices.append({
            'number': r.get('number', ''),
            'itemName': r.get('item_name', ''),
            'qty': int(r.get('qty') or 0),
            'price': float(r.get('price') or 0),
            'customerName': r.get('customer_name') or '',
            'customerEmail': r.get('customer_email') or '',
            'dueDate': r.get('due_date') or '',
            'status': r.get('status') or server.DEFAULT_STATUS,
            'notes': r.get('notes') or '',
        })
server.save_invoices(invoices)

print(f"Migrated {len(items)} items and {len(invoices)} invoices into Postgres.")
