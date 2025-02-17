orphans = set()
senders = set()

with open("graph.txt") as f:
    edges = f.read().split("\n")[2:-1]
    for edge in edges:
        start,end = *edges.split(" ")
        if end not in senders:
            orphans.add(end)
        if start in orphans:
            orphans.remove(start)
            senders.add(start)

print(len(orphans))
