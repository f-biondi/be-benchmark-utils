import os
import sqlite3

os.system("rm sresult.db")
con = sqlite3.connect("sresult.db")
cur = con.cursor()

cur.execute("CREATE TABLE IF NOT EXISTS results(name TEXT, nodes INT, edges INT, time REAL, result INT)")

networks = [
        "alchemy_full",
        "aspirin",
        "benzene",
        "malonaldehyde",
        "ethanol",
        "OVCAR-8",
        "OVCAR-8H",
        "UACC257",
        "ZINC_full",
        "reddit_threads",
        "twitch_egos",

]

for name in networks:
    cur.execute("INSERT INTO results (name, nodes, edges, time, result) VALUES (?,?,?,?,?)", (name, -1, -1, -1, -1))
    con.commit()
