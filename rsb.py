import os
import sqlite3

os.system("rm sresult.db")
con = sqlite3.connect("sresult.db")
cur = con.cursor()

cur.execute("CREATE TABLE IF NOT EXISTS results(name TEXT, time REAL, result INT)")

for n in [100000, 500000, 1000000, 5000000, 10000000]:
    for p in [1, 0.75, 0.5, 0.25, 0]:
        os.system(f"./syns {n} {p}")
        os.system("timeout 7200 python berun.py")
        os.system("pkill java")
        try:
            f = open("res.txt", "r").read().split("\n")
            cur.execute("INSERT INTO results (name, time, result) VALUES (?, ?, ?)", (f"S-{n}-{int(100*p)}", f[1], f[0]))
            con.commit()
            f.close()
            os.system("rm res.txt")
        except FileNotFoundError:
            cur.execute("INSERT INTO results (name, time, result) VALUES (?, ?, ?)", (f"S-{n}-{int(100*p)}", 7200, 0))
            con.commit()
