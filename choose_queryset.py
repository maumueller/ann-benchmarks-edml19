import sys

fn = sys.argv[1]

estimates = []

with open(fn) as f:
    for line in f:
        try:
            i, lid = line.strip().split()
            estimates.append((int(i), float(lid)))
        except:
            pass


estimates.sort(key=lambda x: x[-1])
easy = estimates[:10000]
middle = estimates[len(estimates) // 2 - 5000:len(estimates) // 2 + 5000]
hard = estimates[-10000:]

print("Easiest with avg lid: %f" % (sum(map(lambda x: x[-1], easy)) / 10000))
print(list(map(lambda x: x[0], easy)))
print("Average with avg lid: %f" % (sum(map(lambda x: x[-1], middle)) / 10000))
print(list(map(lambda x: x[0], middle)))
print("Most difficult with avg lid: %f" % (sum(map(lambda x: x[-1], hard)) / 10000))
print(list(map(lambda x: x[0], hard)))
