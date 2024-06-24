# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilitary functions for multiprocessing
# --------------------------------------------------------
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count


def parallel_threads(function, args, workers=0, star_args=False, kw_args=False, front_num=1, Pool=ThreadPool, **tqdm_kw):
    """ tqdm but with parallel execution.

    Will essentially return 
      res = [ function(arg) # default
              function(*arg) # if star_args is True
              function(**arg) # if kw_args is True
              for arg in args]

    Note:
        the <front_num> first elements of args will not be parallelized. 
        This can be useful for debugging.
    """
    while workers <= 0:
        workers += cpu_count()
    if workers == 1:
        front_num = float('inf')

    # convert into an iterable
    try:
        n_args_parallel = len(args) - front_num
    except TypeError:
        n_args_parallel = None
    args = iter(args)

    # sequential execution first
    front = []
    while len(front) < front_num:
        try:
            a = next(args)
        except StopIteration:
            return front  # end of the iterable
        front.append(function(*a) if star_args else function(**a) if kw_args else function(a))

    # then parallel execution
    out = []
    with Pool(workers) as pool:
        # Pass the elements of args into function
        if star_args:
            futures = pool.imap(starcall, [(function, a) for a in args])
        elif kw_args:
            futures = pool.imap(starstarcall, [(function, a) for a in args])
        else:
            futures = pool.imap(function, args)
        # Print out the progress as tasks complete
        for f in tqdm(futures, total=n_args_parallel, **tqdm_kw):
            out.append(f)
    return front + out


def parallel_processes(*args, **kwargs):
    """ Same as parallel_threads, with processes
    """
    import multiprocessing as mp
    kwargs['Pool'] = mp.Pool
    return parallel_threads(*args, **kwargs)


def starcall(args):
    """ convenient wrapper for Process.Pool """
    function, args = args
    return function(*args)


def starstarcall(args):
    """ convenient wrapper for Process.Pool """
    function, args = args
    return function(**args)
