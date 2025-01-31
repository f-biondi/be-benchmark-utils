import os
import sqlite3

os.system("rm sresult.db")
con = sqlite3.connect("sresult.db")
cur = con.cursor()

cur.execute("CREATE TABLE IF NOT EXISTS results(name TEXT, nodes INT, p REAL, time REAL, result INT)")

dbs = {
        "S_100_Thousand_": 100000,
        "S_500_Thousand_": 500000,
        "S_1_Million_": 1000000,
        "S_5_Millions_": 5000000,
        "S_10_Millions_": 10000000,
}

for k,v in dbs.items():
    for p in [1, 0.75, 0.5, 0.25, 0]:
        cur.execute("INSERT INTO results (name, nodes, p, time, result) VALUES (?,?,?,?,?)", (f"{k}{int(100*p)}", v, p, -1, -1))
        con.commit()
