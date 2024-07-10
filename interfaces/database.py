#!/usr/bin/python
import sqlite3


def connect_to_db():
    conn = sqlite3.connect('database/database.db')
    return conn

def insert(query, params):
    id = None
    try:
        conn = connect_to_db()
        cur = conn.cursor()
        cur.execute(query, params)
        conn.commit()
        id = cur.lastrowid
    except:
        conn().rollback()

    finally:
        conn.close()

    return id

def select(query, params = ()):
    result = None
    try:
        conn = connect_to_db()
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(query, params)
        result = cur.fetchall()
    except:
        result = None

    finally:
        conn.close()

    return result

def select_one(query, params = ()):
    result = None
    try:
        conn = connect_to_db()
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(query, params)
        result = cur.fetchone()
    except:
        result = None

    finally:
        conn.close()

    return result

def update(query, params):
    try:
        conn = connect_to_db()
        cur = conn.cursor()
        cur.execute(query, params)
        conn.commit()

        return True
    except:
        conn.rollback()

        return False

    finally:
        conn.close()