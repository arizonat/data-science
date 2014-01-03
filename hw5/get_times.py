from dateutil import parser as dp
import json
from collections import Counter

#FILE="first.json"
FILE="tweets.json"

c = Counter()

l = 0

with open(FILE) as f:
    for line in f:
        j = json.loads(line)
        tim = dp.parse(j["created_at"])
        d = tim.date()
        c[d] += 1

        if l % 500000 == 0:
            print str(l)
        l += 1

f = open("date_counts.csv","w")
for date,i in iter(sorted(c.iteritems())):
    f.write("{},{}\n".format(str(date),str(i)))
f.close()
