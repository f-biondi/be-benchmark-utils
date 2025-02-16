import os
import sqlite3

con = sqlite3.connect("stats.db")
con.row_factory = sqlite3.Row
cur = con.cursor()
cur.execute('select * from results where its = -1;')
result = cur.fetchall()

for r in result:
    print(dict(r))
    #os.system(f"./syns {r['nodes']} {r['p']} {r['degree']}")
    os.system(f"./tunget {r['name']}")

    os.system("timeout 10800 python3 berun.py")
    os.system("pkill java")
    try:
        f = open("/tmp/its.txt", "r")
        l = f.read().split("\n")
        f.close()
        print(r[0], l[1], l[2], l[3])
        cur.execute("UPDATE results SET its = ?, max = ?, min = ?, mean=? where name= ?", (l[0], l[1], l[2], l[3],  r['name']))
        con.commit()
        os.system("rm res.txt")
    except FileNotFoundError:
        print(r['name'], "TIMEOUT")
    os.system("rm graph.txt")
    os.system("rm partition.txt")
