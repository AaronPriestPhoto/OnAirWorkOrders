#!/usr/bin/env python3
"""Check plane specifications."""

import sqlite3

def main():
    conn = sqlite3.connect('onair_jobs.db')
    
    print("Antonov AN-225-210 specs:")
    row = conn.execute('SELECT * FROM plane_specs WHERE plane_type = ?', ('Antonov AN-225-210',)).fetchone()
    if row:
        columns = [d[0] for d in conn.execute("PRAGMA table_info(plane_specs)").fetchall()]
        for col, val in zip(columns, row):
            print(f"  {col}: {val}")
    else:
        print("  No specs found")
    
    conn.close()

if __name__ == "__main__":
    main()
