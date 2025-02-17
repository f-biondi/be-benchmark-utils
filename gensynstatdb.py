import os
import sqlite3

os.system("rm sstat.db")
con = sqlite3.connect("sstat.db")
cur = con.cursor()

cur.execute("CREATE TABLE IF NOT EXISTS results(name TEXT, nodes INT, p REAL, its INT, max INT, min INT, mean INT)")

graphs = {
        100000: "S_100_Thousand_",
        500000: "S_500_Thousand_",
}
for nodes,name in graphs.items():
    for p in [1,0.75,0.5,0.25,0]:
        if nodes > 100000 and p < 0.75:
            break
        cur.execute("INSERT INTO results (name, nodes, p, its, max, min, mean) VALUES (?,?,?,?,?,?,?)", (name+str(int(p*100)), nodes, p, -1, -1, -1, -1))
        con.commit()
