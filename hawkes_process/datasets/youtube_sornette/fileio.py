#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from time import time
from pathlib import Path
import os, os.path
import numpy as np
from numpy import random
# from excited_util import to_rng, uuid_source
from zlib import crc32
import pickle
import warnings
from .test_sets import MEXICAN_SINGER

pd.set_option('io.hdf.default_format', 'table')

inf = float("inf")

class RawYoutube:
    def __init__(self, base_path, *args, **kwargs):
        self.base_path = Path(base_path)
        self.raw_path = self.base_path / 'raw.h5'

    def get_raw_store(self, mode="r"):
        return get_store(str(self.raw_path), mode=mode)

    def get_raw_series(self, video_id=MEXICAN_SINGER):
        with self.get_raw_store(mode="r") as store:
            query = "video_id=={!r}".format(video_id)
            return (
                store.select("tseries", query),
                store.select("video_meta", query)
            )
#
# def shardhash_from_filename(filename, nc=2):
#     return filename[2:2+nc]
# def filename_from_shardhash(shard):
#     return "G_" + shard + ".h5"
# def is_valid_shard_filename(filename, nc=2):
#     return filename.startswith("G_") and filename.endswith(".h5") and len(filename)==(nc+5)
#
# def gen_shardhash(video_id, nc=2):
#     """convert video_id to shardgroup hash by taking two chars of hex digest.
#     crc32 is faster than md5 by a factor of 10^3,
#     which is significant with this many rows.
#     (adler32 was even faster but wasn't distributing the series evenly.)
#     """
#     return hex(crc32(video_id) & 0xffffffff)[-nc:]
#
#
def get_store(fn, mode="r"):
    """
    Open an HDF5 file by absolute path without messing around with trying
    to find the right shard.
    Set sensible compression etc.
    """
    #If we chose "write" then nuke what was already there.
    #this didn't seem to work for a while otherwise
    basedir = os.path.dirname(fn)
    if not os.path.exists(basedir):
        os.makedirs(basedir)

    if mode=="w" and os.path.exists(fn):
        warnings.warn("deleting", fn)
        os.unlink(fn)
    return pd.HDFStore(
        fn,
        mode=mode,
        complevel=5,
        complib="blosc",
        chunksize=2<<18
    )
#
# def get_store_by_path_chunks(fn_chunks, mode="r", **kwargs):
#     """
#     Open an HDF5 file by path chunks, because i am sick of typing
#     os.path.join
#     """
#     fn = os.path.join(*fn_chunks)
#     return get_store(fn, mode=mode, **kwargs)
#
# def get_store_2(pdir=DATA_DIR_ANALYSIS,
#         fn="def",
#         mode="r",
#         **kwargs):
#     """
#     Open an HDF5 file by path chunks, because i am sick of typing
#     os.path.join
#     """
#     return get_store_by_path_chunks([pdir, fn + ".h5"], mode=mode, **kwargs)
#
# def get_result_2(pdir=DATA_DIR_ANALYSIS,
#         fn="def",
#         hd5path="res",
#         columns=None,
#         **kwargs):
#     """
#     Open an HDF5 file by path chunks, because i am sick of typing
#     os.path.join
#     """
#     with get_store_2(pdir, fn, mode="r", **kwargs) as st:
#         return st.select(hd5path, columns=columns)
#
# def get_shard(basedir=DATA_DIR_COOKED, shard="00", mode="r"):
#     """
#     Open an HDF5 file by shard path
#     """
#     fn = os.path.join(basedir, filename_from_shardhash(shard))
#     return get_store_by_path_chunks(
#         [basedir, filename_from_shardhash(shard)], mode=mode
#     )
#
# def get_all_shards(basedir=DATA_DIR_COOKED, include_store=True, nc=2):
#     for filename in os.listdir(basedir):
#         if not is_valid_shard_filename(filename, nc): continue
#         shardhash = shardhash_from_filename(filename, nc)
#         if include_store:
#             yield shardhash, get_shard(basedir, shardhash, mode="r")
#         else:
#             yield shardhash
#
# def get_shard_by_vid(
#         basedir=DATA_DIR_COOKED,
#         video_id=MEXICAN_SINGER,
#         mode="r", nc=2):
#     """
#     Open an HDF5 file by shard path
#     """
#     shard = gen_shardhash(video_id, nc)
#     return get_store_by_path_chunks(
#         [basedir, filename_from_shardhash(shard)], mode=mode
#     )
#
# def get_one_series(basedir=DATA_DIR_COOKED, video_id=MEXICAN_SINGER,
#         transform_fn=None, nc=2):
#     """
#     >>> ts, vm = fileio.get_one_series(_settings.DATA_DIR_CHOPPED, "-2IXE5DcWzg")
#     """
#     shard = gen_shardhash(video_id, nc)
#     if transform_fn is None:
#         transform_fn = lambda t, m: (t, m)
#     with get_shard(basedir, shard) as store:
#         query = "video_id=={!r}".format(video_id)
#         return transform_fn(
#             store.select("tseries", query),
#             store.select("video_meta", query))
#
# def iter_series_from_shard(store, transform_fn=None, limit=inf, nc=2, **kwargs):
#     i = 0
#     if transform_fn is None:
#         transform_fn = lambda t, m, **kwargs: (t, m)
#     for vid, tseries in store['tseries'].groupby(["video_id"], sort=False):
#         query = "video_id=={!r}".format(vid)
#         video_meta = store.select("video_meta", query).iloc[0]
#         result = transform_fn(tseries, video_meta, **kwargs)
#         if result is not None:
#             yield result
#             i +=1
#             if i >= limit:
#                 raise StopIteration
#
# def iter_series_from_dir(basedir=DATA_DIR_COOKED, transform_fn=None, limit=inf, nc=2, **kwargs):
#     i = 0
#     if transform_fn is None:
#         transform_fn = lambda t, m: (t, m)
#     for shardhash, store in get_all_shards(basedir, include_store=True):
#         with store:
#             for result in iter_series_from_shard(
#                     store, transform_fn=fransform_fn, nc=nc, **kwargs):
#                 yield result
#                 i +=1
#                 if i >= limit:
#                     raise StopIteration
#
# def get_concatenated_shards_table(
#         basedir=DATA_DIR_COOKED,
#         hd5path="video_meta",
#         columns=None,
#         nc=2,
#         query=None,
#         ignore_index=True):
#     """return an in-memory table synthesized from the pieces"""
#     shardlist = []
#     for item in get_all_shards(basedir, include_store=True, nc=nc):
#         with item[1] as shard:
#             shardlist.append(shard.select(hd5path, query, columns=columns))
#     return pd.concat(shardlist, ignore_index=ignore_index)
#
# def get_all_video_ids():
#     """return an array of all video ids using cached index"""
#     return get_concatenated_shards_table(DATA_DIR_IDX,
#         hd5path="idx",
#         columns=["video_id"])
#
# def find_in_index(vid=MEXICAN_SINGER, columns=None, with_text=False):
#     columns = [
#         u'id_hash',
#         u'n_samples',
#         u'start_time',
#         u'end_time',
#         u'count_inc_mean',
#         u'count_inc_std',
#         u'time_inc_mean',
#         u'time_inc_std',
#         u'rate_mean',
#         u'rate_std',
#         u'time_span',
#         u'start_count',
#         u'count_span',
#         u'c05_sample',
#         u'c10_sample',
#         u'c25_sample',
#         u'c50_sample',
#         u'c75_sample',
#         u'c90_sample',
#         u'c95_sample',
#         u'upload_time',
#         u'length'
#     ]
#
#     if with_text:
#         columns.extend([u'author', u'title', u'channel'])
#
#     with get_shard_by_vid(basedir=DATA_DIR_IDX, video_id=vid) as store:
#         return store.select('idx',"index==vid & columns=columns").iloc[0,:]
#
# def get_index_table(columns=None, with_text=False,query=None):
#     """return an in-memory table synthesized from the pieces"""
#     if columns is None:
#         columns = [
#             u'id_hash',
#             u'n_samples',
#             u'start_time',
#             u'end_time',
#             u'count_inc_mean',
#             u'count_inc_std',
#             u'time_inc_mean',
#             u'time_inc_std',
#             u'rate_mean',
#             u'rate_std',
#             u'time_span',
#             u'start_count',
#             u'count_span',
#             u'c05_sample',
#             u'c10_sample',
#             u'c25_sample',
#             u'c50_sample',
#             u'c75_sample',
#             u'c90_sample',
#             u'c95_sample',
#             u'upload_time',
#             u'length'
#         ]
#     if with_text:
#         columns.extend([u'author', u'title', u'channel'])
#
#     idx = get_concatenated_shards_table(
#         DATA_DIR_IDX, "idx",
#         columns=columns,
#         ignore_index=False,
#         query=query)
#     # now stored in the index, but persists somehow as a column
#     #idx.drop(["video_id"],inplace=True,axis=1)
#     return idx
#
# def get_description_table(columns=None):
#     """return an in-memory table synthesized from the pieces"""
#     return get_result_2(
#         DATA_DIR_ANALYSIS, fn="description_idx", columns=columns)
#
# def map_shards(
#         pool=None,
#         base_dir=DATA_DIR_CHOPPED,
#         process_fn=None, nc=2,
#         *args, **kwargs):
#     """
#     run the given store-level function on all the stores in a dir.
#     """
#     if pool is None:
#         from parallel import dummy_pool
#         pool = dummy_pool
#     shard_list = get_all_shards(base_dir, include_store=False, nc=nc)
#     pool.map(process_fn, shard_list, *args, **kwargs)
#     return pool
#
# def reduce_shards(
#         base_dir=DATA_DIR_CHOPPED,
#         reduce_fn=None, nc=2,
#         *args, **kwargs):
#     """
#     put the shards back together
#     """
#     if reduce_fn is None:
#         reduce_fn = lambda x: x
#     shard_list = get_all_shards(base_dir, include_store=False, nc=nc)
#     return reduce_fn(shard_list, *args, **kwargs)
#
# def reduce_pool(
#         map_pool_key,
#         reduce_fn=None,
#         *args, **kwargs):
#     """
#     run the given store-level function on all the stores in a dir.
#     """
#     if reduce_fn is None:
#         reduce_fn = lambda x: x
#     from parallel import TrackingPool
#     pool = TrackingPool(pool_key=map_pool_key)
#     file_list = pool.all_files(abs_p=True)
#     return reduce_fn(file_list, *args, **kwargs)
#
# def purge_pool(map_pool_key):
#     reduce_pool(map_pool_key, lambda p: os.unlink(p))
#
# def get_concatenated_pool_table(
#         map_pool_key,
#         hd5path="video_meta",
#         columns=None,
#         base_dir=DATA_DIR_JOB_OUTPUT):
#     """
#     for playing concatenated pool
#     """
#     from parallel import TrackingPool
#
#     pool = TrackingPool(pool_key=map_pool_key)
#     file_list = pool.all_files(abs_p=True, base_dir=base_dir)
#     table_list = []
#     for filepath in file_list:
#         with pd.HDFStore(filepath, mode="r") as store:
#             try:
#                 table_list.append(store.select(hd5path, columns=columns))
#             except KeyError, e:
#                 warnings.warn(
#                     "no matching object {hd5path!r} in {filepath!r}".format(
#                         hd5path=hd5path,
#                         filepath=filepath,
#                     )
#                 )
#
#     return pd.concat(table_list, ignore_index=True)
#
# def store_concatenated_pool_table(
#         map_pool_key,
#         hd5path="res",
#         out_hd5path=None,
#         mode="a",
#         base_dir_in=DATA_DIR_JOB_OUTPUT,
#         base_dir_out=DATA_DIR_ANALYSIS,
#         ):
#     if out_hd5path is None:
#         out_hd5path = hd5path
#     c_table = get_concatenated_pool_table(
#         map_pool_key, hd5path=hd5path,
#         base_dir=base_dir_in)
#     with get_store_by_path_chunks([
#             base_dir_out,
#             map_pool_key + ".h5"
#             ], mode=mode) as st:
#         st.append(out_hd5path, c_table,
#             data_columns=True, #index all
#             index=True, #We can index these because they are atomic
#         )
#     return c_table
#
# def store_concatenated_pool_table_on_disk(
#         map_pool_key,
#         hd5path="video_meta",
#         out_hd5path=None,
#         mode="a",
#         base_dir_in=DATA_DIR_JOB_OUTPUT,
#         base_dir_out=DATA_DIR_ANALYSIS,
#         ):
#     """low-memory version"""
#     if out_hd5path is None:
#         out_hd5path = hd5path
#         pool = TrackingPool(pool_key=map_pool_key)
#         file_list = pool.all_files(abs_p=True, base_dir=base_dir_in)
#     with get_store_by_path_chunks([
#             base_dir_out, map_pool_key + ".h5"], mode=mode) as write_store:
#         for filepath in file_list:
#             with pd.HDFStore(filepath, mode="r") as read_store:
#                 try:
#                     write_store.append(out_hd5path,
#                         read_store.select(hd5path, columns=columns),
#                         data_columns=True, #index all
#                         index=False,
#                     )
#                 except KeyError, e:
#                     warnings.warn(
#                         "no matching object {hd5path!r} in {filepath!r}".format(
#                             hd5path=hd5path,
#                             filepath=filepath,
#                         )
#                     )
#         write_store.create_table_index(out_hd5path, optlevel=9, kind='full')
#     #we just return the store; the table is by assumption too big.
#     c_store = get_store_by_path_chunks([
#             base_dir_out, map_pool_key + ".h5"], mode="a")
#     return c_store
#
# def decorate(frame, index=None):
#     """refresh fit table metadata with info from the index"""
#     if index is None:
#         index = get_index_table()
#     index_cols = index.columns
#     frame_cols = frame.columns
#     wanted_frame_cols = frame_cols.difference(index_cols)
#     frame = frame[wanted_frame_cols]
#     frame = pd.merge(
#         frame, index,
#         left_on=["video_id"],
#         right_index=True,
#         copy=False,
#         how="left",
#         sort=False,
#     )
#     return frame
