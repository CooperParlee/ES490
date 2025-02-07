import numpy as np
import random

m = np.zeros([2, 10]);

for x in range(10):
    for y in range(2):
        m[y, x] = random.randint(1,15);

mavg=np.zeros(m.shape[1])

print(m);

for i in range(m.shape[1]):
    mavg[i] = (m[0, i] + m[1, i])/2

print(mavg);
