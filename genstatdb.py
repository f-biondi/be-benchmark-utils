import os
import sqlite3

os.system("rm stat.db")
con = sqlite3.connect("stat.db")
cur = con.cursor()

cur.execute("CREATE TABLE IF NOT EXISTS results(name TEXT, its INT, max INT, min INT, mean INT)")

networks = [
        "alchemy_full",
        "aspirin",
        "OVCAR-8",
        "OVCAR-8H",
        "UACC257",
        "ZINC_full",
        "reddit_threads",
        "twitch_egos",
]

for name in networks:
    cur.execute("INSERT INTO results (name, its, max, min, mean) VALUES (?,?,?,?,?)", (name, -1, -1, -1, -1))
    con.commit()
