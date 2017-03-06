#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from time import time
import numpy as np
from fileio import get_raw_store, get_shard, gen_shardhash, get_all_shards,\
    filename_from_shardhash, iter_series_from_shard, map_shards, get_description_table, get_one_series
from sys import maxint
from scipy.interpolate import interp1d
import test_sets

from parallel import dummy_pool

import os


def chop(nc=2):
    """
    slice big table into small tables
    """
    # For estimating operation times
    mean_time = 100.0
    time_now = 0.0
    time_then = 0.0

    with get_raw_store("r") as raw_store:
        # May be able to retrieve upload time using multiple table queries (JOIN-lite)
        # http://pandas.pydata.org/pandas-docs/version/0.15.1/io.html#multiple-table-queries

        # #Can I skip this step now and just hard-code in a value?
        # video_ids = np.asarray(
        #     raw_store.select('video_meta', columns=['video_id']),
        #     dtype=np.dtype((str, 12)) #otherwise object
        # ).flatten() #otherwise it is 2d
        # n_vids = video_ids.shape[0]
        n_vids = 6970136

        # Nuke existing chopped tables
        for shardhash in get_all_shards(DATA_DIR_CHOPPED,
                include_store=False, nc=nc):
            fn = os.path.join(DATA_DIR_CHOPPED, filename_from_shardhash(shardhash))
            print "removing", fn
            os.unlink(fn)

        # A question of note is what characters are used in labels.
        # I know there are newlines (suspicious, surely). Are there others?
        # >>> used_chars=set(list("".join(video_ids)))
        # Gives
        # >>> "".join(sorted(list(used_chars)))
        # '\n-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz'
        # That's 65 characters, including newlines, mostly safe ASCII.
        # \n is the only nonprintable
        # It's a bad nonprintable though, because it breaks the parse engine.
        # exemplar "6fTl-2B3qjg\n"

        n_tseries_rows = raw_store.get_node("tseries").table.nrows
        n_meta_rows = raw_store.get_node("video_meta").table.nrows
        # (hopefully) overestimate number of rows in shardtables
        est_tseries_shardtable_size = 2*n_tseries_rows/256
        est_meta_shardtable_size = 2*n_meta_rows/256

        print "choppping tseries data"

        time_now = time()

        i=0
        for chunk in raw_store.select('tseries', chunksize=CHUNKSIZE):
            print i*CHUNKSIZE, n_tseries_rows, mean_time * (
                n_tseries_rows/float(CHUNKSIZE)-i)
            chunk['video_id'] = chunk['video_id'].str.replace("\n", "~")
            chunk_shardhash = chunk['video_id'].apply(gen_shardhash)
            print "hashed!"
            for shardhash, grp in chunk.groupby(chunk_shardhash, sort=False):
                print chunk_shardhash, grp.shape
                with get_shard(DATA_DIR_CHOPPED, shardhash, "a"
                        ) as chopped_store:
                    chopped_store.append('tseries',
                        grp.dropna(),
                        expectedrows=est_tseries_shardtable_size,
                        data_columns=['video_id'],
                        min_itemsize={'video_id':12},
                        index=False, #suppress index
                    )
            time_then = time_now
            time_now = time()
            i += 1
            mean_time = 0.95 * mean_time + 0.05 * (time_now-time_then)

        #We chunk the metadata operations differently:
        # fewer rows so we can optimally partition in memory before disk IO.
        video_meta = raw_store["video_meta"]
        video_meta['video_id'] = video_meta['video_id'].str.replace("\n", "~")
        video_meta_shardhash = video_meta['video_id'].apply(gen_shardhash)
        #Actually video_meta is basically useless. I should make my own.
        i = 0
        for shardhash, grp in video_meta.groupby(
                video_meta_shardhash,
                sort=False):
            print i, shardhash
            with get_shard(DATA_DIR_CHOPPED, shardhash, "a") as chopped_store:
                chopped_store.append(
                    'video_meta',
                    grp,
                    expectedrows=est_meta_shardtable_size,
                    data_columns=True,  # index all
                    min_itemsize={'video_id':12},
                    index=True, #We can index these because they are atomic
                )
            i += 1
    # later, do indexing, perhaps with ptrepack
    # http://stackoverflow.com/a/17898006/11730
    for shardhash, store in get_all_shards(DATA_DIR_CHOPPED):
        print "indexing", shardhash
        with store:
            store.create_table_index('tseries',
            columns=["video_id"],
            optlevel=9,
            kind='full')

def index_reduce(pool=dummy_pool):
    pass

def index(pool=dummy_pool, *args, **kwargs):
    "shortcut to remind me how this thing works"
    map_shards(pool=pool,
        base_dir=DATA_DIR_CHOPPED,
        process_fn=index_shard,
        *args, **kwargs)

def index_shard(shardhash, limit=maxint,
        debug=False,
        *args, **kwargs):
    """index the videos by tricky metadata"""
    if debug: print "indexing", shardhash
    start_time = time()
    with get_shard(DATA_DIR_CHOPPED, shardhash, mode="r") as store:
        index_list = list(iter_series_from_shard(
            store,
            transform_fn=index_for_frame,
            limit=limit,
            debug=debug,
            **kwargs))
    index_table = pd.DataFrame(np.concatenate(index_list))
    desc_table = get_description_table()
    index_table = pd.merge(
        index_table, desc_table,
        on=["video_id"], copy=False,
        how="left",
        sort=False,
    )
    index_table.index = index_table.video_id
    index_table.drop(["video_id"],inplace=True,axis=1)
    with get_shard(DATA_DIR_IDX, shardhash, mode="w") as store:
        store.append('idx',
            index_table,
            min_itemsize={'video_id':12},
            data_columns=True)
    print time() - start_time

def index_for_vid(vid=test_sets.MEXICAN_SINGER,
        *args, **kwargs):
    ts, vm = get_one_series(DATA_DIR_CHOPPED, vid)
    return index_for_frame(ts, vm, *args, **kwargs)

def index_for_frame(tseries_grp, video_meta_rec, min_n_samples=100,
        debug=False,
        *args, **kwargs):
    """
    indexed version of the everything
    """
    tseries_grp, video_meta_rec = spiced_frame(tseries_grp, video_meta_rec)
    n_samples = tseries_grp.shape[0]
    if n_samples<min_n_samples:
        return None
    vid = tseries_grp.iloc[0,0]
    if debug: print vid

    idx_rec = np.zeros((1,), dtype=[
        ('video_id', 'S12'),
        ('id_hash', 'S8'),
        ('n_samples', 'i4'),
        ('start_time', 'f4'),
        ('end_time', 'f4'),
        ('count_inc_mean', 'f4'),
        ('count_inc_std', 'f4'),
        ('time_inc_mean', 'f4'),
        ('time_inc_std', 'f4'),
        ('rate_mean', 'f4'),
        ('rate_std', 'f4'),
        ('time_span', 'f4'),
        ('start_count', 'i4'),
        ('count_span', 'i4'),
        ('c05_sample', 'f4'),
        ('c10_sample', 'f4'),
        ('c25_sample', 'f4'),
        ('c50_sample', 'f4'),
        ('c75_sample', 'f4'),
        ('c90_sample', 'f4'),
        ('c95_sample', 'f4'),
    ])
    idx_rec['video_id'] = vid
    idx_rec['id_hash'] = gen_shardhash(vid, 8)
    idx_rec['n_samples'] = n_samples

    start_count = tseries_grp['view_count'].iloc[0]
    idx_rec['start_count'] = tseries_grp['view_count'].iloc[0]
    start_time = tseries_grp['run_time'].iloc[0]
    idx_rec['start_time'] = start_time
    end_time = tseries_grp['run_time'].iloc[-1]
    idx_rec['end_time'] = end_time
    count_span = tseries_grp['view_count'].iloc[-1] - start_count
    idx_rec['count_span'] = count_span
    time_span = end_time - start_time
    idx_rec["time_span"] = time_span
    idx_rec['count_inc_mean'] = tseries_grp['view_count_diff'].mean()
    idx_rec['count_inc_std'] = tseries_grp['view_count_diff'].std()
    idx_rec['time_inc_mean'] = tseries_grp['run_time_diff'].mean()
    idx_rec['time_inc_std'] = tseries_grp['run_time_diff'].std()
    idx_rec['rate_mean'] = tseries_grp['rate'].mean()
    idx_rec['rate_std'] = tseries_grp['rate'].std()

    #Very coarse quantile calculation
    # Should ideally be bootstrapped.
    rel_count = tseries_grp['view_count'].values - start_count
    rel_time = tseries_grp['run_time'].values - start_time
    count_to_time = interp1d(rel_count, rel_time)
    rel_time_quantiles = count_to_time(
        np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]) * count_span
    )/time_span

    idx_rec['c05_sample'] = rel_time_quantiles[0]
    idx_rec['c10_sample'] = rel_time_quantiles[1]
    idx_rec['c25_sample'] = rel_time_quantiles[2]
    idx_rec['c50_sample'] = rel_time_quantiles[3]
    idx_rec['c75_sample'] = rel_time_quantiles[4]
    idx_rec['c90_sample'] = rel_time_quantiles[5]
    idx_rec['c95_sample'] = rel_time_quantiles[6]

    return idx_rec

def trimmed_frame(tseries_grp, video_meta_rec,
        trim_meta=True,
        *args, **kwargs):
    """
    Trim out nonsense steps from the data,
    """
    tseries_grp.sort('run_time', inplace=True)
    view_count = pd.expanding_max(
        tseries_grp['view_count'].astype('float')).astype('int')
    mask = np.ediff1d(view_count.values , to_begin=1)>0
    # watch out for duplicate metadata
    if trim_meta:
        if len(video_meta_rec.shape)>1:
            video_meta_rec = video_meta_rec.iloc[0,:]

    return tseries_grp.iloc[mask].dropna(), video_meta_rec.dropna()

def spiced_frame(tseries_grp, video_meta_rec,
        trim=True, time_scale=3600.0*24.0,
        *args, **kwargs):
    """
    Optionally trim out nonsense steps from the data,
    rescale for the sake of a convenient order of magnitude
    decorate with useful columns
    """
    vid = tseries_grp['video_id'].iloc[0]
    if trim:
        tseries_grp, video_meta_rec = trimmed_frame(tseries_grp, video_meta_rec)
        if "fold" in video_meta_rec:
            del(video_meta_rec["fold"])
    # squash
    tseries_grp = pd.concat([tseries_grp], ignore_index=True)
    # index by date
    tseries_grp_index = pd.to_datetime(
        tseries_grp.run_time, unit='s')

    n_steps = tseries_grp.shape[0]
    tseries_grp = tseries_grp.sort(["run_time"])
    ## rescale run_time to be approximately 1 day
    run_time = tseries_grp["run_time"]/time_scale
    view_count = tseries_grp["view_count"]
    run_time_diff = run_time.diff()
    view_count_diff = view_count.diff()
    rate = view_count_diff/run_time_diff
    rate_diff = rate.diff()
    start_time = run_time.iloc[0]
    end_time = run_time.iloc[-1]
    start_count = view_count.iloc[0]
    end_count = view_count.iloc[-1]

    spiced_tseries_frame = pd.DataFrame.from_items([
        ('video_id', vid),
        ('run_time', run_time.astype('float32')),
        ('run_time_diff', run_time_diff.astype('float32')),
        ('view_count', view_count),
        ('view_count_diff', view_count_diff),
        ('rate', rate.astype('float32')),
        ('rate_diff', rate_diff.astype('float32')),
    ], orient="columns")
    spiced_tseries_frame.index = index=tseries_grp_index
    spiced_video_meta = video_meta_rec.copy().dropna()
    spiced_video_meta["usable_steps"] = n_steps
    spiced_video_meta["start_time"] = start_time
    spiced_video_meta["end_time"] = end_time
    spiced_video_meta["start_count"] = start_count
    spiced_video_meta["end_count"] = end_count
    spiced_video_meta["time_span"] = end_time - start_time
    spiced_video_meta["view_count_span"] = end_count - start_count
    spiced_video_meta["rate_std"] = rate.std()
    spiced_video_meta["rate_diff_std"] = rate_diff.std()
    # remove artefacts of processing with confusing names
    if "run_time" in spiced_video_meta:
        del(spiced_video_meta["run_time"])
    if "view_count" in spiced_video_meta:
        del(spiced_video_meta["view_count"])
    if "fold" in spiced_video_meta:
        del(spiced_video_meta["fold"])

    return spiced_tseries_frame, spiced_video_meta

def cook(pool=dummy_pool, min_n_samples=100, *args, **kwargs):
    map_shards(pool=pool,
        base_dir=DATA_DIR_CHOPPED,
        process_fn=cook_shard,
        *args, **kwargs)

def cook_shard(shardhash, min_n_samples=100,
        limit=maxint, *args, **kwargs):

    print "cooking", shardhash
    ts_list = []
    vm_list = []
    with get_shard(DATA_DIR_CHOPPED, shardhash, mode="r") as store:
        for ts, vm in iter_series_from_shard(
                store,
                transform_fn=trimmed_frame,
                limit=limit):
            if ts.shape[0]>=min_n_samples:
                ts_list.append(ts)
                vm_list.append(vm)
    vm_table = pd.DataFrame.from_records(vm_list)
    ts_table = pd.concat(ts_list, ignore_index=True)
    with get_shard(DATA_DIR_COOKED, shardhash, mode="w") as store:
        store.append('video_meta',
            vm_table,
            min_itemsize={'video_id':12},
            data_columns=True)
        store.append('tseries',
            ts_table,
            min_itemsize={'video_id':12},
            data_columns=['video_id'])
