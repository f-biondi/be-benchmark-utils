import os
import sqlite3

con = sqlite3.connect("sresult.db")
con.row_factory = sqlite3.Row
cur = con.cursor()
cur.execute('select * from results where result = -1 order by nodes;')
result = cur.fetchall()

for r in result:
    print(dict(r))
    os.system(f"cp synthetic-hg/graph_{r['nodes']}_{int(r['p']*100)}_{r['degree']}.txt graph.txt")
    os.system(f"cp synthetic-hg/partition_{r['nodes']}_{int(r['p']*100)}_{r['degree']}.txt partition.txt")
    os.system("timeout 7200 python berun.py")
    os.system("pkill java")
    try:
        f = open("res.txt", "r")
        l = f.read().split("\n")
        f.close()
        print(r['name'], l[1], l[0])
        cur.execute("UPDATE results SET time = ?, result = ? where name= ?", (l[1], l[0], r['name']))
        con.commit()
        os.system("rm res.txt")
    except FileNotFoundError:
        print(r['name'], "TIMEOUT")
        cur.execute("UPDATE results SET time = ?, result = ? where name= ?", (7200, r['nodes'], r['name']))
        con.commit()
    os.system("rm graph.txt")
    os.system("rm partition.txt")
