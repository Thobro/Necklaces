import math
import random
P = [(1, 0), (1.4, 0), (1.7, 1), (2.2, 2), (2.3, 0), (4.5, 0), (4.6, 1), (4.7, 2)] # n points
C = [4, 5, 4, 2, 3] # m colors

n = 1000
m = 50
P = [(random.uniform(0, 30000), random.randint(0, m - 1)) for k in range(n)]
C = [random.randint(1, 8) for k in range(m)]
S = [0 for k in range(m)]

p_left = 0
p_right = 0
min_int = (math.inf, 0, 0)

#TODO: At least 0 points initialise

P = sorted(P, key=lambda p: p[0]) # O(n log n)
print("Sweep")
s = 0
while p_right != len(P): # O(n)
    if s < m:
        S[P[p_right][1]] += 1
        if S[P[p_right][1]] == C[P[p_right][1]]:
            s += 1
            if s == m:
                p_right -= 1
        p_right += 1
    else:
        if P[p_right][0] - P[p_left][0] < min_int[0]:
            min_int = (P[p_right][0] - P[p_left][0], p_left, p_right)
            print(min_int)

        # Can we move the left pointer right?
        if S[P[p_left][1]] - 1 >= C[P[p_left][1]] and p_left < p_right:
            S[P[p_left][1]] -= 1
            p_left += 1
        else: # If not, we move the right pointer right
            p_right += 1
            if p_right != len(P):
                S[P[p_right][1]] += 1

print("Brute force") # O(n^2 * k)
min_int = (math.inf, 0, 0)
for l in range(len(P)):
    for r in range(l, len(P)):
        S = [0 for k in range(len(C))]
        for k in range(l, r+1):
            S[P[k][1]] += 1
        if all([S[i] >= C[i] for i in range(len(S))]):
            if P[r][0] - P[l][0] < min_int[0]:
                min_int = (P[r][0] - P[l][0], l, r)
                print(min_int)
