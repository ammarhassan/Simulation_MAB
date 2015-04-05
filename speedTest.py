from scipy.stats import lognorm
from datetime import datetime

iterations = 200000
l = [(x+1.0)/(iterations) for x in range(iterations)]

t1 = datetime.now()

for i in l:
	lognorm(1, loc=0).pdf(i)

t2 = datetime.now()
print t2 - t1

a = lognorm(1, loc=0).pdf(l)

print datetime.now() - t2
print "length", len(a)

