"""Split–merge moves for BoolMF.

Gibbs updates change one variable at a time, so they cannot leave states such as two components
merged into one, a component that has swallowed a smaller one living in a subset of its carriers,
or one component split in two by carriers: every single-variable step out of these states lowers
the posterior before the whole move would raise it. These Metropolis–Hastings moves change two
components at once, following the restricted Gibbs split–merge sampler of Jain & Neal (2004)
adapted to overlapping Boolean components.

Two slots a and b define a *union*: the samples active in either (rows) and the features that
belong to either (columns). Every row and column of the union is in a only, b only, or both,
written as the options (1, 0), (0, 1) and (1, 1); nothing outside the union changes.

* split: a used slot and an empty slot; the union is the used slot. Propose a random allocation.
* merge: two used slots; propose a = union, b = empty.
* reallocate: two used slots; propose a new allocation of the same union.

Proposals come from a restricted Gibbs scan started at a launch state. The launch depends only
on the union, the data and random numbers, never on the current allocation, so the same launch
gives the probability of the reverse proposal. The target is the posterior with the component
probabilities pi and rho integrated out (Beta–Bernoulli), given the rates and alpha; the chain
redraws pi and rho from their conditionals right after these moves, which keeps the joint
posterior invariant.

The block likelihood has two forms. ``mode = 0`` (count tables, both likelihoods with rates
shared by components): the state is the count matrix C and an entry's log-likelihood is
T[row, count]. ``mode = 1`` (log-survival, ``noisy_or`` with per-component rates): the state is
L[i, j] = sum over covering components of log(1 - lambda_ik), each component adds S[i, k] =
log(1 - lambda_ik), and log P(x = 0) = log(1 - background_i) + L. In both forms a block entry's
value is ``base + za * sa + zb * sb`` with per-row increments sa, sb (1 for counts).
"""

import math

import numpy as np
from numba import njit

from .kernels import _next_uniform

SPLIT, MERGE, REALLOCATE, FACTOR, UNFACTOR = 0, 1, 2, 3, 4
MOVE_NAMES = ("split", "merge", "reallocate", "factor", "unfactor")
N_MOVES = 5


@njit(cache=True)
def _log_beta_bernoulli(m, a, b, N):
    """log P(a given column with m ones out of N) under Bernoulli(p), p ~ Beta(a, b)."""
    return (math.lgamma(a + m) + math.lgamma(b + N - m) - math.lgamma(a + b + N)
            - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)))


@njit(cache=True)
def _entry_table_counts(Vb, base, rows, sa_r, sb_r, T1, T0):
    R, M = Vb.shape
    rs = 1 if T1.shape[0] > 1 else 0
    E = np.zeros((R, M, 4))
    for r in range(R):
        g = rows[r] * rs
        for m in range(M):
            v = Vb[r, m]
            if v < 0:
                continue
            x0 = base[r, m]
            c0 = int(x0 + 0.5)
            ca = int(x0 + sa_r[r] + 0.5)
            cb = int(x0 + sb_r[r] + 0.5)
            cab = int(x0 + sa_r[r] + sb_r[r] + 0.5)
            if v == 1:
                E[r, m, 0] = T1[g, c0]
                E[r, m, 1] = T1[g, ca]
                E[r, m, 2] = T1[g, cb]
                E[r, m, 3] = T1[g, cab]
            else:
                E[r, m, 0] = T0[g, c0]
                E[r, m, 1] = T0[g, ca]
                E[r, m, 2] = T0[g, cb]
                E[r, m, 3] = T0[g, cab]
    return E


@njit(inline="always")
def _ls_ll(v, x):
    """Log-likelihood of v given log P(x = 0) = x."""
    if v == 1:
        if x < -1e-12:
            return math.log(-math.expm1(x))
        return -27.631021115928547                # log(1e-12)
    return x


@njit(cache=True)
def _entry_table_logsurv(Vb, base, rows, sa_r, sb_r, LB0):
    R, M = Vb.shape
    rsb = 1 if LB0.shape[0] > 1 else 0
    E = np.zeros((R, M, 4))
    for r in range(R):
        lb = LB0[rows[r] * rsb]
        for m in range(M):
            v = Vb[r, m]
            if v < 0:
                continue
            x0 = lb + base[r, m]
            E[r, m, 0] = _ls_ll(v, x0)
            E[r, m, 1] = _ls_ll(v, x0 + sa_r[r])
            E[r, m, 2] = _ls_ll(v, x0 + sb_r[r])
            E[r, m, 3] = _ls_ll(v, x0 + sa_r[r] + sb_r[r])
    return E


@njit(cache=True)
def block_entry_table(mode, Vb, base, rows, sa_r, sb_r, T1, T0, LB0):
    """E[r, m, c]: log-likelihood of block entry (r, m) when it is covered by slot a (bit 0 of
    c) and/or slot b (bit 1), everything else fixed; 0 for missing entries."""
    if mode == 0:
        return _entry_table_counts(Vb, base, rows, sa_r, sb_r, T1, T0)
    return _entry_table_logsurv(Vb, base, rows, sa_r, sb_r, LB0)


@njit(cache=True)
def block_log_target(E, za, zb, ua, ub, zpa, zpb, n, upa, upb, F):
    """Log posterior (up to a constant shared by all allocations of the same union)."""
    R, M = E.shape[0], E.shape[1]
    s = 0.0
    for r in range(R):
        for m in range(M):
            s += E[r, m, za[r] * ua[m] + 2 * zb[r] * ub[m]]
    ma = 0
    mb = 0
    for r in range(R):
        ma += za[r]
        mb += zb[r]
    sa = 0
    sb = 0
    for m in range(M):
        sa += ua[m]
        sb += ub[m]
    s += _log_beta_bernoulli(ma, zpa, zpb, n) + _log_beta_bernoulli(mb, zpa, zpb, n)
    s += _log_beta_bernoulli(sa, upa, upb, F) + _log_beta_bernoulli(sb, upa, upb, F)
    return s


@njit(cache=True)
def _pick(w0, w1, w2, target, state, use_target):
    """Choose an option with probabilities proportional to exp(w); return (option, log p, state)."""
    mx = max(w0, max(w1, w2))
    p0 = math.exp(w0 - mx)
    p1 = math.exp(w1 - mx)
    p2 = math.exp(w2 - mx)
    tot = p0 + p1 + p2
    if use_target:
        o = target
    else:
        state, u = _next_uniform(state)
        x = u * tot
        o = 0 if x < p0 else (1 if x < p0 + p1 else 2)
    p = p0 if o == 0 else (p1 if o == 1 else p2)
    return o, math.log(p / tot), state


@njit(cache=True)
def _option(a, b):
    return 0 if b == 0 else (1 if a == 0 else 2)


@njit(cache=True)
def restricted_scan(E, za, zb, ua, ub, zpa, zpb, n, upa, upb, F, order_r, order_m, state,
                    use_target, tza, tzb, tua, tub):
    """One restricted Gibbs scan over the union: columns in ``order_m``, then rows in ``order_r``.

    Each element is redrawn from its conditional among (1, 0), (0, 1), (1, 1), or, with
    ``use_target``, set to the target's value. Updates za, zb, ua, ub in place and returns
    (state, log probability of the choices made). E is ``block_entry_table``.
    """
    R, M = E.shape[0], E.shape[1]
    logq = 0.0
    sa = 0
    sb = 0
    for m in range(M):
        sa += ua[m]
        sb += ub[m]
    ma = 0
    mb = 0
    for r in range(R):
        ma += za[r]
        mb += zb[r]
    for t in range(M):
        m = order_m[t]
        sa -= ua[m]
        sb -= ub[m]
        la1 = math.log(upa + sa)
        la0 = math.log(upb + F - 1 - sa)
        lb1 = math.log(upa + sb)
        lb0 = math.log(upb + F - 1 - sb)
        w0 = la1 + lb0
        w1 = la0 + lb1
        w2 = la1 + lb1
        for r in range(R):
            ia = za[r]
            ib = 2 * zb[r]
            w0 += E[r, m, ia]
            w1 += E[r, m, ib]
            w2 += E[r, m, ia + ib]
        o, lp, state = _pick(w0, w1, w2, _option(tua[m], tub[m]), state, use_target)
        logq += lp
        ua[m] = 1 if o != 1 else 0
        ub[m] = 1 if o != 0 else 0
        sa += ua[m]
        sb += ub[m]
    for t in range(R):
        r = order_r[t]
        ma -= za[r]
        mb -= zb[r]
        la1 = math.log(zpa + ma)
        la0 = math.log(zpb + n - 1 - ma)
        lb1 = math.log(zpa + mb)
        lb0 = math.log(zpb + n - 1 - mb)
        w0 = la1 + lb0
        w1 = la0 + lb1
        w2 = la1 + lb1
        for m in range(M):
            ia = ua[m]
            ib = 2 * ub[m]
            w0 += E[r, m, ia]
            w1 += E[r, m, ib]
            w2 += E[r, m, ia + ib]
        o, lp, state = _pick(w0, w1, w2, _option(tza[r], tzb[r]), state, use_target)
        logq += lp
        za[r] = 1 if o != 1 else 0
        zb[r] = 1 if o != 0 else 0
        ma += za[r]
        mb += zb[r]
    return state, logq


@njit(cache=True)
def _randint(state, m):
    """Uniform integer in [0, m)."""
    state, u = _next_uniform(state)
    k = int(u * m)
    return state, (k if k < m else m - 1)


@njit(cache=True)
def _permutation(state, m):
    p = np.arange(m)
    for i in range(m - 1, 0, -1):
        state, j = _randint(state, i + 1)
        tmp = p[i]
        p[i] = p[j]
        p[j] = tmp
    return state, p


@njit(cache=True)
def _any(x):
    for v in x:
        if v != 0:
            return True
    return False


PARTNER_EPS = 0.02          # weight floor so that every pair of used slots can be proposed


@njit(cache=True)
def _slot_lists(nz, nu, used, empty):
    """Fill ``used`` (members and carriers) and ``empty`` (neither); return their sizes."""
    Nu = 0
    Ne = 0
    for t in range(nz.size):
        if nz[t] > 0 and nu[t] > 0:
            used[Nu] = t
            Nu += 1
        elif nz[t] == 0 and nu[t] == 0:
            empty[Ne] = t
            Ne += 1
    return Nu, Ne


@njit(cache=True)
def _partner_weights(Z, U, free_idx, nz, nu, used, Nu, t1):
    """Weight of each used slot as a partner of slot t1: Jaccard similarity of carriers plus
    Jaccard similarity of members, plus a small floor (zero for t1 itself)."""
    n = Z.shape[0]
    F = U.shape[0]
    k1 = free_idx[t1]
    ic = np.zeros(Nu, np.int64)
    im = np.zeros(Nu, np.int64)
    for i in range(n):
        if Z[i, k1] != 0:
            for x in range(Nu):
                ic[x] += Z[i, free_idx[used[x]]]
    for j in range(F):
        if U[j, k1] != 0:
            for x in range(Nu):
                im[x] += U[j, free_idx[used[x]]]
    w = np.zeros(Nu)
    for x in range(Nu):
        t = used[x]
        if t == t1:
            continue
        w[x] = (PARTNER_EPS + ic[x] / (nz[t1] + nz[t] - ic[x])
                + im[x] / (nu[t1] + nu[t] - im[x]))
    return w


@njit(cache=True)
def _log_partner_prob(Z, U, free_idx, nz, nu, used, Nu, t1, t2):
    w = _partner_weights(Z, U, free_idx, nz, nu, used, Nu, t1)
    tot = 0.0
    w2 = 0.0
    for x in range(Nu):
        tot += w[x]
        if used[x] == t2:
            w2 = w[x]
    return math.log(w2 / tot)


@njit(cache=True)
def _write(Z, U, rows, cols, k1, k2, za, zb, ua, ub):
    for r in range(rows.size):
        Z[rows[r], k1] = za[r]
        Z[rows[r], k2] = zb[r]
    for m in range(cols.size):
        U[cols[m], k1] = ua[m]
        U[cols[m], k2] = ub[m]


@njit(cache=True)
def _count(x):
    s = 0
    for v in x:
        s += v
    return s


@njit(cache=True)
def split_merge_kernel(mode, V, Z, U, STATE, T1, T0, S, LB0, free_idx, zpa, zpb, upa, upb,
                       n_attempts, n_launch, seed, stats):
    """``n_attempts`` proposals on the free slots, in place (STATE = C or L, see module doc).

    Move selection: the first slot is uniform over used slots; a merge or reallocation partner
    is drawn with weight proportional to its similarity to the first slot (``_partner_weights``);
    a split partner is uniform over empty slots. Selection probabilities are recomputed in the
    proposed state for the reverse move.
    """
    n, F = V.shape
    nf = free_idx.size
    if nf < 2:
        return
    log_moves = math.log(N_MOVES)
    state = np.uint64(seed)
    nz = np.zeros(nf, np.int64)
    nu = np.zeros(nf, np.int64)
    for t in range(nf):
        k = free_idx[t]
        for i in range(n):
            nz[t] += Z[i, k]
        for j in range(F):
            nu[t] += U[j, k]
    used = np.empty(nf, np.int64)
    empty = np.empty(nf, np.int64)
    for _ in range(n_attempts):
        Nu, Ne = _slot_lists(nz, nu, used, empty)
        state, move = _randint(state, N_MOVES)
        if move == SPLIT:
            if Nu < 1 or Ne < 1:
                continue
            state, x = _randint(state, Nu)
            t1 = used[x]
            state, x = _randint(state, Ne)
            t2 = empty[x]
            log_sel_fwd = -log_moves - math.log(Nu) - math.log(Ne)
        else:
            if Nu < 2:
                continue
            state, x = _randint(state, Nu)
            t1 = used[x]
            w = _partner_weights(Z, U, free_idx, nz, nu, used, Nu, t1)
            tot = w.sum()
            state, u = _next_uniform(state)
            u *= tot
            y = 0
            acc = w[0]
            while acc <= u and y < Nu - 1:
                y += 1
                acc += w[y]
            if w[y] == 0.0:                    # guard against rounding onto t1 itself
                continue
            t2 = used[y]
            log_sel_fwd = -log_moves - math.log(Nu) + math.log(w[y] / tot)
        k1 = free_idx[t1]
        k2 = free_idx[t2]
        if move <= REALLOCATE:
            stats[0, move] += 1

        # ---- the union and its block ----------------------------------------------------
        rows = np.empty(nz[t1] + nz[t2], np.int64)
        R = 0
        for i in range(n):
            if Z[i, k1] != 0 or Z[i, k2] != 0:
                rows[R] = i
                R += 1
        rows = rows[:R]
        cols = np.empty(nu[t1] + nu[t2], np.int64)
        M = 0
        for j in range(F):
            if U[j, k1] != 0 or U[j, k2] != 0:
                cols[M] = j
                M += 1
        cols = cols[:M]
        za0 = np.empty(R, np.int8)
        zb0 = np.empty(R, np.int8)
        ua0 = np.empty(M, np.int8)
        ub0 = np.empty(M, np.int8)
        for r in range(R):
            za0[r] = Z[rows[r], k1]
            zb0[r] = Z[rows[r], k2]
        for m in range(M):
            ua0[m] = U[cols[m], k1]
            ub0[m] = U[cols[m], k2]
        sa_r = np.ones(R)
        sb_r = np.ones(R)
        if mode == 1:
            rss = 1 if S.shape[0] > 1 else 0
            for r in range(R):
                sa_r[r] = S[rows[r] * rss, k1]
                sb_r[r] = S[rows[r] * rss, k2]
        Vb = np.empty((R, M), np.int8)
        base = np.empty((R, M))
        for r in range(R):
            for m in range(M):
                Vb[r, m] = V[rows[r], cols[m]]
                base[r, m] = (STATE[rows[r], cols[m]] - za0[r] * ua0[m] * sa_r[r]
                              - zb0[r] * ub0[m] * sb_r[r])
        E = block_entry_table(mode, Vb, base, rows, sa_r, sb_r, T1, T0, LB0)

        if move >= FACTOR:
            # Rewrites that keep which entries are covered. Two shapes of a pair (a, b):
            #   P: a's members a strict subset of b's; some carriers of a not in b;
            #   Q: b's carriers a strict subset of a's; some members of b not in a.
            # factor (P -> Q): a takes b's carriers too, b keeps only the members a lacks.
            # unfactor (Q -> P): the inverse. Overlaps that do not change coverage (carriers in
            # both slots in P, members in both slots in Q) are lost by one map and redrawn by
            # the other, independently with probabilities p_D and p_E; these are functions of
            # counts that both shapes share, so the reverse proposal probability is exact.
            a_only_rows = 0
            b_only_cols = 0
            for r in range(R):
                if za0[r] != 0 and zb0[r] == 0:
                    a_only_rows += 1
            for m in range(M):
                if ub0[m] != 0 and ua0[m] == 0:
                    b_only_cols += 1
            p_d = min(0.5, max(1e-3, (1.0 + a_only_rows) / (2.0 + n)))
            p_e = min(0.5, max(1e-3, (1.0 + b_only_cols) / (2.0 + F)))
            ok = a_only_rows > 0 and b_only_cols > 0
            za1 = np.empty(R, np.int8)
            zb1 = zb0.copy()
            ua1 = ua0.copy()
            ub1 = np.empty(M, np.int8)
            log_q = 0.0
            if move == FACTOR:
                for m in range(M):
                    if ua0[m] != 0 and ub0[m] == 0:
                        ok = False
                if not ok:
                    continue
                for r in range(R):
                    za1[r] = 1
                    if zb0[r] != 0:            # reverse redraws the overlap D
                        log_q += math.log(p_d) if za0[r] != 0 else math.log1p(-p_d)
                for m in range(M):
                    if ua0[m] != 0:            # draw the overlap E
                        state, u = _next_uniform(state)
                        e = u < p_e
                        ub1[m] = 1 if e else 0
                        log_q -= math.log(p_e) if e else math.log1p(-p_e)
                    else:
                        ub1[m] = ub0[m]
            else:
                for r in range(R):
                    if zb0[r] != 0 and za0[r] == 0:
                        ok = False
                if not ok:
                    continue
                for r in range(R):
                    if zb0[r] != 0:            # draw the overlap D
                        state, u = _next_uniform(state)
                        d = u < p_d
                        za1[r] = 1 if d else 0
                        log_q -= math.log(p_d) if d else math.log1p(-p_d)
                    else:
                        za1[r] = za0[r]
                for m in range(M):
                    ub1[m] = 1 if (ua0[m] != 0 or ub0[m] != 0) else 0
                    if ua0[m] != 0:            # reverse redraws the overlap E
                        log_q += math.log(p_e) if ub0[m] != 0 else math.log1p(-p_e)
            stats[0, move] += 1
        else:
            # ---- launch state: depends on the union, the data and random numbers only --------
            if M >= 2:
                state, j1 = _randint(state, M)
                state, j2 = _randint(state, M - 1)
                if j2 >= j1:
                    j2 += 1
            else:
                j1 = 0
                j2 = 0
            za = np.empty(R, np.int8)
            zb = np.empty(R, np.int8)
            for r in range(R):
                za[r] = 1 if Vb[r, j1] == 1 else 0
                zb[r] = 1 if Vb[r, j2] == 1 else 0
                if za[r] == 0 and zb[r] == 0:
                    state, o = _randint(state, 3)
                    za[r] = 1 if o != 1 else 0
                    zb[r] = 1 if o != 0 else 0
            ua = np.ones(M, np.int8)
            ub = np.ones(M, np.int8)
            for _s in range(n_launch):
                state, pr = _permutation(state, R)
                state, pm = _permutation(state, M)
                state, _lq = restricted_scan(E, za, zb, ua, ub, zpa, zpb, n, upa, upb, F, pr,
                                             pm, state, False, za0, zb0, ua0, ub0)
            state, order_r = _permutation(state, R)
            state, order_m = _permutation(state, M)

            # ---- proposal and the probability of the reverse proposal -------------------------
            if move == MERGE:
                _st, logq_rev = restricted_scan(E, za, zb, ua, ub, zpa, zpb, n, upa, upb, F,
                                                order_r, order_m, state, True,
                                                za0, zb0, ua0, ub0)
                za1 = np.ones(R, np.int8)
                zb1 = np.zeros(R, np.int8)
                ua1 = np.ones(M, np.int8)
                ub1 = np.zeros(M, np.int8)
                log_q = logq_rev
            else:
                za1 = za.copy()
                zb1 = zb.copy()
                ua1 = ua.copy()
                ub1 = ub.copy()
                state, logq_fwd = restricted_scan(E, za1, zb1, ua1, ub1, zpa, zpb, n, upa,
                                                  upb, F, order_r, order_m, state, False,
                                                  za0, zb0, ua0, ub0)
                if not (_any(za1) and _any(zb1) and _any(ua1) and _any(ub1)):
                    continue                       # one slot would be left empty: reject
                if move == SPLIT:
                    log_q = -logq_fwd
                else:
                    _st, logq_rev = restricted_scan(E, za, zb, ua, ub, zpa, zpb, n, upa, upb, F,
                                                    order_r, order_m, state, True,
                                                    za0, zb0, ua0, ub0)
                    log_q = logq_rev - logq_fwd

        # write the proposal; the reverse selection probability is computed in that state
        _write(Z, U, rows, cols, k1, k2, za1, zb1, ua1, ub1)
        nz1_old = nz[t1]
        nz2_old = nz[t2]
        nu1_old = nu[t1]
        nu2_old = nu[t2]
        nz[t1] = _count(za1)
        nz[t2] = _count(zb1)
        nu[t1] = _count(ua1)
        nu[t2] = _count(ub1)
        if move == MERGE:
            log_sel_rev = -log_moves - math.log(Nu - 1) - math.log(Ne + 1)
        else:
            Nu2, _Ne2 = _slot_lists(nz, nu, used, empty)
            log_sel_rev = (-log_moves - math.log(Nu2)
                           + _log_partner_prob(Z, U, free_idx, nz, nu, used, Nu2, t1, t2))
        log_r = (block_log_target(E, za1, zb1, ua1, ub1, zpa, zpb, n, upa, upb, F)
                 - block_log_target(E, za0, zb0, ua0, ub0, zpa, zpb, n, upa, upb, F)
                 + log_sel_rev - log_sel_fwd + log_q)
        state, u = _next_uniform(state)
        if log_r >= 0.0 or u < math.exp(log_r):
            for r in range(R):
                for m in range(M):
                    val = base[r, m] + za1[r] * ua1[m] * sa_r[r] + zb1[r] * ub1[m] * sb_r[r]
                    if mode == 0:
                        STATE[rows[r], cols[m]] = int(val + 0.5)
                    else:
                        STATE[rows[r], cols[m]] = val
            stats[1, move] += 1
        else:
            _write(Z, U, rows, cols, k1, k2, za0, zb0, ua0, ub0)
            nz[t1] = nz1_old
            nz[t2] = nz2_old
            nu[t1] = nu1_old
            nu[t2] = nu2_old


def split_merge_moves(V, Z, U, C, T1, T0, free, zprior, uprior, n_attempts, n_launch, rng,
                      stats=None):
    """Attempt ``n_attempts`` split, merge, reallocate, factor or unfactor moves, in place.

    Count form: C holds the counts, T1, T0 the likelihood tables, one row per sample or one row
    (or a 1-D table) shared by all samples. zprior = (a, b) of the Beta prior on activation
    probabilities (alpha / K and 1 under the Indian buffet process); uprior = (a, b) of the Beta
    prior on membership probabilities. ``stats`` (optional, int64 array of shape (2, N_MOVES))
    accumulates attempts and acceptances per move type, in the order of ``MOVE_NAMES``.
    """
    if T1.ndim == 1:
        T1, T0 = T1[None, :], T0[None, :]
    _run(0, V, Z, U, C, T1, T0, _DUMMY2, _DUMMY1, free, zprior, uprior, n_attempts, n_launch,
         rng, stats)


def split_merge_moves_logsurv(V, Z, U, L, S, LB0, free, zprior, uprior, n_attempts, n_launch,
                              rng, stats=None):
    """Log-survival form (``noisy_or`` with per-component rates): L[i, j] = sum over covering
    components of S[i, k] = log(1 - lambda_ik); LB0 = log(1 - background), one entry per sample
    or one shared. S has one row per sample or one shared row."""
    _run(1, V, Z, U, L, _DUMMY2, _DUMMY2, S, LB0, free, zprior, uprior, n_attempts, n_launch,
         rng, stats)


_DUMMY1 = np.zeros(1)
_DUMMY2 = np.zeros((1, 1))


def _run(mode, V, Z, U, STATE, T1, T0, S, LB0, free, zprior, uprior, n_attempts, n_launch, rng,
         stats):
    if stats is None:
        stats = np.zeros((2, N_MOVES), np.int64)
    split_merge_kernel(mode, V, Z, U, STATE, T1, T0, S, np.ascontiguousarray(LB0, float),
                       np.flatnonzero(free).astype(np.int64), float(zprior[0]),
                       float(zprior[1]), float(uprior[0]), float(uprior[1]), int(n_attempts),
                       int(n_launch), np.uint64(rng.integers(0, 2**63 - 1)), stats)
