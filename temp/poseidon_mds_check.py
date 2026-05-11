#!/usr/bin/env python3
"""
Pure-Python port of the three Poseidon MDS security checks
(algorithm_1, algorithm_2, algorithm_3) from
    extgit.iaik.tugraz.at/krypto/hadeshash
    code/generate_parameters_grain.sage

Applied to the 8x8 circulant matrix with first row
    [-1, -2, 1, -2, 8, 8, 4, 1]
over the Goldilocks prime field p = 2^64 - 2^32 + 1.

Convention: s = 1 (one partial S-box per partial round), matching Poseidon.
"""

import random
from itertools import combinations

random.seed(1)

P = (1 << 64) - (1 << 32) + 1
NUM_CELLS = 8
FIRST_ROW = [-1, -2, 1, -2, 8, 8, 4, 1]

# ------- modular & matrix helpers -------

def mat_mul(A, B):
    rA, cA, cB = len(A), len(A[0]), len(B[0])
    out = [[0]*cB for _ in range(rA)]
    for i in range(rA):
        Ai = A[i]
        Oi = out[i]
        for k in range(cA):
            a = Ai[k]
            if a == 0: continue
            Bk = B[k]
            for j in range(cB):
                Oi[j] = (Oi[j] + a * Bk[j]) % P
    return out

def mat_pow(M, e):
    n = len(M)
    R = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    B = [row[:] for row in M]
    while e > 0:
        if e & 1:
            R = mat_mul(R, B)
        B = mat_mul(B, B)
        e >>= 1
    return R

def mat_sub(A, B):
    return [[(A[i][j] - B[i][j]) % P for j in range(len(A[0]))] for i in range(len(A))]

def mat_apply(M, v):
    return [sum(M[i][j] * v[j] for j in range(len(v))) % P for i in range(len(M))]

def scalar_eye(n, c):
    return [[c if i == j else 0 for j in range(n)] for i in range(n)]

def is_zero_mat(A):
    return all(all(x == 0 for x in row) for row in A)

# ------- RREF & subspace ops -------

def rref(rows):
    if not rows:
        return []
    M = [list(r)[:] for r in rows]
    ncols = len(M[0])
    p = 0
    for col in range(ncols):
        sel = None
        for r in range(p, len(M)):
            if M[r][col] % P != 0:
                sel = r; break
        if sel is None:
            continue
        M[p], M[sel] = M[sel], M[p]
        inv = pow(M[p][col] % P, -1, P)
        M[p] = [(x * inv) % P for x in M[p]]
        for r in range(len(M)):
            if r == p: continue
            f = M[r][col] % P
            if f != 0:
                M[r] = [(M[r][i] - f * M[p][i]) % P for i in range(ncols)]
        p += 1
        if p == len(M): break
    return [tuple(r) for r in M if any(x % P != 0 for x in r)]

def subspace_dim(S):
    return len(S)

def subspace_apply(M, S, t):
    return rref([tuple(mat_apply(M, list(v))) for v in S])

def subspace_intersect(A, B, n):
    """Zassenhaus algorithm: basis of A ∩ B."""
    if not A or not B:
        return []
    rows = []
    for v in A: rows.append(list(v) + list(v))
    for v in B: rows.append(list(v) + [0]*n)
    R = rref(rows)
    out = []
    for row in R:
        if all(x % P == 0 for x in row[:n]):
            tail = tuple(row[n:])
            if any(x % P != 0 for x in tail):
                out.append(tail)
    return rref(out)

def right_kernel(M):
    """Basis of {x : M x = 0}. M is list-of-rows, vectors live in F^{len(M[0])}."""
    if not M:
        return []
    n = len(M[0])
    R = rref([list(r) for r in M])
    pivot_cols = []
    for row in R:
        for c, val in enumerate(row):
            if val % P != 0:
                pivot_cols.append(c); break
    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(n) if c not in pivot_set]
    basis = []
    for f in free_cols:
        v = [0] * n
        v[f] = 1
        for i, pc in enumerate(pivot_cols):
            v[pc] = (-R[i][f]) % P
        basis.append(tuple(v))
    return rref(basis)

def full_space_basis(t):
    return rref([tuple(1 if j == i else 0 for j in range(t)) for i in range(t)])

# ------- polynomial ops over F_p (constant-term first) -------

def poly_strip(p):
    p = list(p)
    while len(p) > 1 and p[-1] % P == 0:
        p.pop()
    return p

def poly_sub(a, b):
    n = max(len(a), len(b))
    r = [0]*n
    for i in range(n):
        if i < len(a): r[i] = (r[i] + a[i]) % P
        if i < len(b): r[i] = (r[i] - b[i]) % P
    return poly_strip(r)

def poly_mul(a, b):
    r = [0]*(len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x % P == 0: continue
        for j, y in enumerate(b):
            r[i+j] = (r[i+j] + x*y) % P
    return poly_strip(r)

def poly_divmod(a, b):
    a = list(a); b = poly_strip(list(b))
    if len(b) == 1 and b[0] % P == 0:
        raise ZeroDivisionError
    if len(a) < len(b):
        return [0], poly_strip(a)
    q = [0]*(len(a) - len(b) + 1)
    inv_lead = pow(b[-1] % P, -1, P)
    while len(a) >= len(b):
        coeff = (a[-1] * inv_lead) % P
        deg = len(a) - len(b)
        q[deg] = coeff
        for i in range(len(b)):
            a[deg + i] = (a[deg + i] - coeff * b[i]) % P
        while len(a) > 1 and a[-1] % P == 0:
            a.pop()
        if len(a) == 1 and a[0] % P == 0:
            break
    return poly_strip(q), poly_strip(a)

def poly_mod(a, b): return poly_divmod(a, b)[1]

def poly_gcd(a, b):
    a = poly_strip(list(a)); b = poly_strip(list(b))
    while not (len(b) == 1 and b[0] % P == 0):
        a, b = b, poly_mod(a, b)
    if len(a) == 1 and a[0] % P == 0:
        return [0]
    inv = pow(a[-1] % P, -1, P)
    return [(c * inv) % P for c in a]

def poly_pow_mod(base, exp, m):
    result = [1]
    base = poly_mod(base, m)
    while exp > 0:
        if exp & 1:
            result = poly_mod(poly_mul(result, base), m)
        base = poly_mod(poly_mul(base, base), m)
        exp >>= 1
    return result

def char_poly(M):
    """Faddeev-LeVerrier; returns coeffs constant-first."""
    n = len(M)
    c = [0]*(n+1)
    c[0] = 1
    M_prev = [row[:] for row in M]
    c[1] = (-sum(M_prev[i][i] for i in range(n))) % P
    for k in range(2, n+1):
        temp = [row[:] for row in M_prev]
        for i in range(n): temp[i][i] = (temp[i][i] + c[k-1]) % P
        Mk = mat_mul(M, temp)
        tr = sum(Mk[i][i] for i in range(n)) % P
        c[k] = (-tr * pow(k, -1, P)) % P
        M_prev = Mk
    return list(reversed(c))

def find_F_roots(f):
    """Distinct roots of f in F_p (returns list, possibly empty)."""
    f = poly_strip(list(f))
    if len(f) <= 1: return []
    xp = poly_pow_mod([0, 1], P, f)
    g = poly_gcd(f, poly_sub(xp, [0, 1]))
    if len(g) <= 1: return []
    return cz_split(g)

def cz_split(g):
    """g is a product of distinct linear factors over F_p; return roots."""
    if len(g) == 1: return []
    if len(g) == 2:
        # monic x + g[0] -> root -g[0]
        return [(-g[0]) % P]
    while True:
        alpha = random.randrange(0, P)
        h = poly_pow_mod([alpha, 1], (P - 1) // 2, g)
        h = poly_sub(h, [1])
        if len(h) == 1 and h[0] % P == 0:
            continue
        d = poly_gcd(g, h)
        if len(d) <= 1 or len(d) >= len(g):
            continue
        q, _ = poly_divmod(g, d)
        return cz_split(d) + cz_split(q)

# ------- Poseidon checks -------

def generate_vectorspace(round_num, M, M_round, t):
    s = 1
    if round_num == 0:
        return full_space_basis(t)
    if round_num == 1:
        return rref([tuple(1 if j == i else 0 for j in range(t)) for i in range(s, t)])
    rows = []
    for i in range(round_num - 1):
        for j in range(s):
            rows.append(M_round[i][j][s:])  # last t-s entries of row j of M^(i+1)
    if not rows:
        kernel = [tuple(1 if k == idx else 0 for k in range(t-s)) for idx in range(t-s)]
    else:
        kernel = right_kernel([list(r) for r in rows])
    return rref([tuple([0]*s + list(v)) for v in kernel])

def algorithm_1(M, t):
    s = 1
    r = (t - s) // s
    M_round = [mat_pow(M, j+1) for j in range(t+1)]
    for i in range(1, r+1):
        mat_test = mat_pow(M, i)
        entry = mat_test[0][0]
        if is_zero_mat(mat_sub(mat_test, scalar_eye(t, entry))):
            return (False, 1, i)
        S = generate_vectorspace(i, M, M_round, t)
        cp = char_poly(mat_test)
        eigvals = set(find_F_roots(cp))
        basis_vectors = []
        for lam in eigvals:
            eig_subspace = right_kernel(mat_sub(mat_test, scalar_eye(t, lam)))
            basis_vectors.extend(subspace_intersect(S, eig_subspace, t))
        IS = rref(basis_vectors)
        if 1 <= subspace_dim(IS) < t:
            return (False, 2, i)
        for j in range(1, i+1):
            if S == subspace_apply(mat_pow(M, j), S, t):
                return (False, 3, (i, j))
    return (True, 0, None)

def algorithm_2(M, t):
    s = 1
    indices = list(range(0, s))
    powerset = []
    for k in range(1, s+1):
        powerset.extend(combinations(indices, k))
    for I_s in powerset:
        test_next = False
        nb = [tuple(1 if i == l else 0 for i in range(t)) for l in I_s]
        IS = rref(nb)
        nb2 = list(nb)
        for i in range(s, t):
            nb2.append(tuple(1 if k == i else 0 for k in range(t)))
        full_iota = rref(nb2)
        for l in I_s:
            v = tuple(1 if i == l else 0 for i in range(t))
            while True:
                delta = subspace_dim(IS)
                v = tuple(mat_apply(M, list(v)))
                IS = rref(list(IS) + [v])
                if subspace_dim(IS) == t:
                    test_next = True; break
                inter = subspace_intersect(IS, full_iota, t)
                if subspace_dim(inter) != subspace_dim(IS):
                    test_next = True; break
                if subspace_dim(IS) <= delta:
                    break
            if test_next: break
        if test_next: continue
        return (False, (IS, list(I_s)))
    return (True, None)

def algorithm_3(M, t):
    l = 4 * t
    for r in range(2, l + 1):
        if not algorithm_2(mat_pow(M, r), t)[0]:
            return (False, r)
    return (True, None)

# ------- run -------

def build_circulant(first_row):
    n = len(first_row)
    return [[first_row[(j - i) % n] % P for j in range(n)] for i in range(n)]

if __name__ == "__main__":
    M = build_circulant(FIRST_ROW)
    t = NUM_CELLS

    print(f"Matrix: 8x8 circulant, first row {FIRST_ROW}")
    print(f"Field:  Goldilocks p = 2^64 - 2^32 + 1 = {P}")
    print()

    r1 = algorithm_1(M, t)
    print(f"Algorithm 1 (invariant subspace trails, inactive S-boxes): {r1}")
    r2 = algorithm_2(M, t)
    print(f"Algorithm 2 (iterative subspace trails, inactive S-boxes): {r2}")
    r3 = algorithm_3(M, t)
    print(f"Algorithm 3 (iterative subspace trails over M^r, r=2..4t):  {r3}")

    ok = r1[0] and r2[0] and r3[0]
    print()
    print(f"Result: matrix is {'SECURE for Poseidon' if ok else 'NOT SECURE — fails a check'}.")
