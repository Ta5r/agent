import os
import sqlite3
from datetime import datetime

DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "support.db")

def get_db_connection():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create orders table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        order_id TEXT PRIMARY KEY,
        customer_name TEXT NOT NULL,
        customer_email TEXT NOT NULL,
        product_name TEXT NOT NULL,
        status TEXT NOT NULL,
        tracking_number TEXT,
        purchase_date TEXT NOT NULL
    )
    """)
    
    # Create tickets table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tickets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_email TEXT NOT NULL,
        subject TEXT NOT NULL,
        description TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'Open',
        created_at TEXT NOT NULL
    )
    """)
    
    # Check if orders are seeded
    cursor.execute("SELECT COUNT(*) as count FROM orders")
    if cursor.fetchone()["count"] == 0:
        mock_orders = [
            ("ORD-1001", "Alice Smith", "alice@example.com", "Quantum Soundbar Pro", "Delivered", "TRK-982348", "2026-08-15"),
            ("ORD-1002", "Bob Jones", "bob@example.com", "UltraView 4K Projector", "Shipped", "TRK-102938", "2026-08-28"),
            ("ORD-1003", "Charlie Brown", "charlie@example.com", "ErgoDesk Premium", "Processing", None, "2026-08-29"),
            ("ORD-1004", "Diana Prince", "diana@example.com", "Shield Smartwatch", "Delivered", "TRK-473829", "2026-08-10"),
            ("ORD-1005", "Evan Wright", "evan@example.com", "Acoustic Noise-Cancelling Headphones", "Cancelled", None, "2026-08-20")
        ]
        cursor.executemany("""
        INSERT INTO orders (order_id, customer_name, customer_email, product_name, status, tracking_number, purchase_date)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, mock_orders)
        
    conn.commit()
    conn.close()

def get_order(order_id: str, email: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM orders WHERE LOWER(order_id) = LOWER(?) AND LOWER(customer_email) = LOWER(?)",
        (order_id.strip(), email.strip())
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def create_ticket(email: str, subject: str, description: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO tickets (customer_email, subject, description, status, created_at) VALUES (?, ?, ?, 'Open', ?)",
        (email.strip(), subject.strip(), description.strip(), now_str)
    )
    ticket_id = cursor.lastrowid
    conn.commit()
    
    cursor.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row)

def list_orders():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM orders ORDER BY purchase_date DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def list_tickets():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tickets ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# Initialize on import
init_db()
