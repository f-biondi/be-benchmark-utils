import os
import sqlite3
from datetime import datetime

con = sqlite3.connect("sresult.db")
con.row_factory = sqlite3.Row
cur = con.cursor()
cur.execute('select * from results where result = -1 order by nodes;')
result = cur.fetchall()

for r in result:
    print(dict(r))
    #os.system(f"./syns {r['nodes']} {r['p']} {r['degree']}")
    os.system(f"./vcon {r['name']}")

    start = datetime.now()
    os.system(f"timeout 10800 cat graph.txt | ./MDPmin.out > res.txt")
    end = datetime.now()
    try:
        f = open("res.txt", "r")
        result = int(f.read().split("\n")[0].split(" ")[0])
        f.close()
        time = (end - start).total_seconds()
        cur.execute("UPDATE results SET time = ?, result = ? where name= ?", (time, result, r['name']))
        con.commit()
        os.system("rm res.txt")
    except Exception:
        print(r['name'], "TIMEOUT")
        cur.execute("UPDATE results SET time = ?, result = ? where name= ?", (10800, r['nodes'], r['name']))
        con.commit()
    os.system("rm graph.txt")
    os.system("rm partition.txt")
