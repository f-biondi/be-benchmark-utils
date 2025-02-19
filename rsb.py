import os
import sqlite3

con = sqlite3.connect("sresult.db")
con.row_factory = sqlite3.Row
cur = con.cursor()
cur.execute('select * from results where result = -1 order by nodes;')
result = cur.fetchall()

def adapt():
    nodes = 0
    lines = []
    with open("graph.txt") as f:
        lines = f.read().split("\n")[:-1]
        nodes = int(lines[0])
    with open("graph.txt", "w") as f:
        f.write("\n".join(lines[2:]))
    return nodes


for r in result:
    print(dict(r))
    #os.system(f"./syns {r['nodes']} {r['p']} {r['degree']}")
    os.system(f"./tunget {r['name']}")
    nodes = adapt()

    os.system("timeout 10800 java -jar epsBE.jar graph.txt 131  0 0 1 res.txt")
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
        cur.execute("UPDATE results SET time = ?, result = ? where name= ?", (10800, r['nodes'], r['name']))
        con.commit()
    os.system("rm graph.txt")
    os.system("rm partition.txt")
