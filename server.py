"""Small static + JSON API server backing index.html with CSV persistence."""

import csv
import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.abspath(__file__))
ITEMS_CSV = os.path.join(ROOT, 'items.csv')
INVOICES_CSV = os.path.join(ROOT, 'invoices.csv')

ITEMS_HEADER = ['id', 'name', 'qty', 'price']
INVOICES_HEADER = ['number', 'item_name', 'qty', 'price']


def ensure_csv(path, header):
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(header)


def read_csv(path, header):
    ensure_csv(path, header)
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, header, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_items():
    items = []
    for r in read_csv(ITEMS_CSV, ITEMS_HEADER):
        items.append({
            'id': int(r['id']),
            'name': r['name'],
            'qty': int(r['qty']),
            'price': float(r['price']),
        })
    return items


def save_items(items):
    write_csv(ITEMS_CSV, ITEMS_HEADER, items)


def load_invoices():
    invoices = []
    for r in read_csv(INVOICES_CSV, INVOICES_HEADER):
        invoices.append({
            'number': r['number'],
            'itemName': r['item_name'],
            'qty': int(r['qty']),
            'price': float(r['price']),
        })
    return invoices


def save_invoices(invoices):
    rows = [{
        'number': i['number'],
        'item_name': i['itemName'],
        'qty': i['qty'],
        'price': i['price'],
    } for i in invoices]
    write_csv(INVOICES_CSV, INVOICES_HEADER, rows)


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/api/items':
            return self._json(load_items())
        if path == '/api/invoices':
            return self._json(load_invoices())
        return super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode('utf-8') if length else ''
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            return self._error(400, 'invalid json')

        if path == '/api/items':
            return self._create_item(data)
        if path == '/api/invoices':
            return self._create_invoice(data)
        return self._error(404, 'not found')

    def do_DELETE(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        if path == '/api/items':
            try:
                item_id = int(qs.get('id', [''])[0])
            except ValueError:
                return self._error(400, 'invalid id')
            items = load_items()
            remaining = [i for i in items if i['id'] != item_id]
            if len(remaining) == len(items):
                return self._error(404, 'item not found')
            save_items(remaining)
            return self._json({'deleted': item_id})
        if path == '/api/invoices':
            number = qs.get('number', [''])[0].strip()
            if not number:
                return self._error(400, 'invoice number required')
            invoices = load_invoices()
            remaining = [i for i in invoices if i['number'] != number]
            deleted = len(invoices) - len(remaining)
            save_invoices(remaining)
            return self._json({'deleted': deleted, 'number': number})
        return self._error(404, 'not found')

    def _create_item(self, data):
        try:
            name = str(data['name']).strip()
            qty = int(data['qty'])
            price = float(data['price'])
        except (KeyError, ValueError, TypeError):
            return self._error(400, 'invalid item payload')
        if not name or qty < 0 or price < 0:
            return self._error(400, 'invalid item values')

        items = load_items()
        next_id = max((i['id'] for i in items), default=0) + 1
        new_item = {'id': next_id, 'name': name, 'qty': qty, 'price': price}
        items.append(new_item)
        save_items(items)
        return self._json(new_item, status=201)

    def _create_invoice(self, data):
        try:
            number = str(data['number']).strip()
            item_id = int(data['itemId'])
            qty = int(data['qty'])
            price = float(data['price'])
        except (KeyError, ValueError, TypeError):
            return self._error(400, 'invalid invoice payload')
        if not number or qty <= 0 or price < 0:
            return self._error(400, 'invalid invoice values')

        items = load_items()
        item = next((i for i in items if i['id'] == item_id), None)
        if item is None:
            return self._error(400, 'item not found')
        if qty > item['qty']:
            return self._error(400, f'only {item["qty"]} of "{item["name"]}" available')

        item['qty'] -= qty
        save_items(items)

        invoices = load_invoices()
        new_invoice = {'number': number, 'itemName': item['name'], 'qty': qty, 'price': price}
        invoices.append(new_invoice)
        save_invoices(invoices)

        return self._json({'invoice': new_invoice, 'item': item}, status=201)

    def _json(self, data, status=200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, message):
        self._json({'error': message}, status=status)


if __name__ == '__main__':
    os.chdir(ROOT)
    # When PORT is set (e.g. on Render/Fly/Railway) bind publicly; otherwise
    # only bind to localhost for local development.
    port = int(os.environ.get('PORT', '3000'))
    host = '0.0.0.0' if os.environ.get('PORT') else 'localhost'
    display_host = 'localhost' if host == 'localhost' else host
    print(f'Serving http://{display_host}:{port}  (Ctrl+C to stop)')
    HTTPServer((host, port), Handler).serve_forever()
