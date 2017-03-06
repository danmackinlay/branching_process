#!/usr/bin/env python
# -*- coding: utf-8 -*

VOLATILITY_CLUSTERING = '-oAFf-KD5Q4'
LONE_SPIKE = '-jvUZTlzGKw'
FLUCTUATE = '-1WpE25fA2U'
PLATEAU = '-a_lxYqz5Jk'
MULTI_PLATEAUX = '-aPkiofW_Kw'
SURPRISE_PEAK = '-Drg6ezuemY'
MEXICAN_SINGER = '-2IXE5DcWzg'
KRAKOV_HOOLIGANS = 'w_db4nuMg0E' #amazing; counted from 0 organic grwth Cracovia Hooligans (Anty Wisla)
SAMOA_JOE = 'RXPK4H7urOo' #Oct 31 2006 trailer for Nov 19 boxing match
DOLL_FACE = 'zl6hNj1uOkY' #weird HUGE late spike from Björk collborator
OTTER_HANDS = 'epUk3T2Kfno' #Otter video
EARLY_SPIKE = '3cPNV_xZy_8' #Milan football mayhem

SINGLE_REGIME = dict(
    VOLATILITY_CLUSTERING=VOLATILITY_CLUSTERING,
    LONE_SPIKE=LONE_SPIKE,
    FLUCTUATE=FLUCTUATE
)
MULTI_REGIME = dict(
    KRAKOV_HOOLIGANS=KRAKOV_HOOLIGANS,
    PLATEAU=PLATEAU,
    MULTI_PLATEAUX=MULTI_PLATEAUX,
    SURPRISE_PEAK=SURPRISE_PEAK,
    MEXICAN_SINGER=MEXICAN_SINGER,
    DOLL_FACE=DOLL_FACE,
    OTTER_HANDS=OTTER_HANDS,
    SAMOA_JOE=SAMOA_JOE,
    EARLY_SPIKE=EARLY_SPIKE,
)

def get_tricky_cases(min_size=200, index=None, shuffle=False):
    import numpy as np
    if index is None:
        import fileio
        index = fileio.get_index_table()
    s = set(get_flagships(index=index))
    s = s.union(get_lead_balloons(index=index, min_size=min_size, N=300))
    s = s.union(get_dragon_kings_abs(index=index, min_size=min_size))
    s = np.array(list(s))
    if shuffle:
        np.random.shuffle(s)
    return list(s)

def get_flagships(index=None, shuffle=False):
    import numpy as np
    s = np.array(list(set(MULTI_REGIME.values() + SINGLE_REGIME.values())))
    if shuffle:
        np.random.shuffle(s)
    return list(s)

def get_whatever(N=100, min_size=200, index=None):
    import numpy as np
    if index is None:
        import fileio
        index = fileio.get_index_table()
    vids = index[
        (index.n_samples>min_size)
    ].index.values
    return np.random.choice(vids, N, replace=False)

def get_lead_balloons(cutoff_time=0.03,
        quantile="c50",
        min_size=200,
        N=None,
        index=None):
    import numpy as np
    if index is None:
        import fileio
        index = fileio.get_index_table()
    vids = index[
        (index[quantile+"_sample"]<cutoff_time) &
        (index.n_samples>min_size)
    ].index.values
    if N is None:
        return vids
    return np.random.choice(vids, N, replace=False)

#[   1624    3860   20241  179607 1112550 3523227 8001631]
def get_dragon_kings_abs(cutoff_count=1112550, min_size=200, index=None):
    if index is None:
        import fileio
        index = fileio.get_index_table()
    heavy_ids = index[
        (index.count_span>cutoff_count) &
        (index.n_samples>min_size)
    ].index.values
    return heavy_ids

def get_dragon_kings_rel(cutoff_frac=0.75, index=None):
    import numpy as np
    if index is None:
        import fileio
        index = fileio.get_index_table()
    n_occ = index.count_span.values
    n_occ = np.sort(n_occ)
    cum_occ = np.cumsum(n_occ)
    cutoff_count = n_occ[np.searchsorted(cum_occ, cum_occ[-1]*cutoff_frac)]
    return get_dragon_kings_abs(cutoff_count, index=index)
