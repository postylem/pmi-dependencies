def entropy(vec, base=None):
    """Compute entropy of unnormalized probability vector."""
    total = sum(vec)
    probs = [i / total for i in vec]
    entropy = 0.
    base = e if base is None else base
    for p in probs:
        if p < 0:
            ent_p = np.inf
        elif p == 0:
            ent_p = 0.
        else:
            ent_p = p * log(p, base)
        entropy -= ent_p

    return entropy


def peakiness_entropy(vec, abs=false):
    """Measure peakiness as entropy"""
    if len(vec) == 1:
        return np.nan
    if abs:
        vec = [abs(x) for x in vec]
    return 1 - entropy(vec) / log(len(vec))


def peakiness_sparseness(vec):
    """Measure peakiness as sparseness
    as defined in Hoyer 2004"""
    d = len(vec)
    if d == 1:
        return np.nan
    abs_total = sum(abs(x) for x in vec)
    l2 = sqrt(sum(x**2 for x in vec))
    numerator = sqrt(d) - abs_total / l2
    return numerator / (sqrt(d) - 1)
