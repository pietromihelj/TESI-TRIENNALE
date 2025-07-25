import os
import mne
import time
import numpy as np
from functools import partial
import itertools as it
import portion as P
from functools import wraps
from inspect import signature

def check_type(p_name, p_value, allowed_types):
    """"
    p_name: nome del parametro
    p_value: valore del parametro del quale dovro controllare il tipo
    allowed_types: tipi permessi
    """

    if not isinstance(p_name, str):
        raise TypeError("`parameter` must be %s, but got %s" % (repr(str), repr(type(p_name))))
    if not isinstance(allowed_types, list):
        raise TypeError("`allowed_types` must be %s, but got %s" % (repr(list), repr(type(allowed_types))))
    allowed_types = tuple(allowed_types)
    if isinstance(p_value, allowed_types):
        return p_value
    
    msg = ("Invalid type for the '{parameter}' parameter: "
       '{options}, but got {value_type} instead.')

    if len(allowed_types) == 0:
        raise RuntimeError("allowed_types is not set.")
    else:
        options = ', '.join(repr(v) for v in allowed_types)

    raise TypeError(msg.format(parameter=p_name, options=options,
                            value_type=repr(type(p_value))))

def get_path_list(d_path, f_extensions,sub_d = False):
    """"
    d_path: path assoluto della directory
    f_extensions: ezìstensioni permesse
    sub_d: ricorsività nelle sottodirectory
    """
    check_type('d_path',d_path,[str])
    check_type('f_extensions', f_extensions, [list])
    check_type('sub_d', sub_d, [bool])

    d_path = get_abs_path(d_path)
    if not os.path.isdir(d_path):
        raise NotADirectoryError(d_path)
    
    ext_lower = []
    for ext in f_extensions:
            if isinstance(ext, str):
                ext_lower.append(ext.lower())
            else:
                raise TypeError("elements of extension must be string (suffixes of files), "
                                "but got %s" % str(f_extensions))
    f_extensions = ext_lower

    path_list = []
    if sub_d:
        for root,_,files in os.walk(d_path):
            for name in files:
                if accept_file(root, name, f_extensions):
                    path_list.append(os.path.join(root, name))
    else:
        files = os.listdir(d_path)
        for name in files:
            if accept_file(d_path, name, f_extensions):
                path_list.append(os.path.join(d_path, name))
    
    return np.array(path_list)

def get_abs_path(path):
    check_type('path', path, [str])
    return os.path.realpath(os.path.expanduser(os.path.expandvars(path)))

def accept_file(dir, f_name, ext):
    check_type('dir', dir, [str])
    check_type('f_name', f_name, [str])
    check_type('ext', ext, [list])

    f_name = os.path.join(dir, f_name)
    cond_1 = os.path.isfile(f_name)
    cond_2 = os.path.splitext(f_name)[1].lower() in ext
    return cond_1 and cond_2

def get_raw(f_path, preload=False):
    check_type('f_path', f_path, [str])
    if os.path.isdir(f_path):
        raise KeyError('can`t identify %s.'%f_path)
    return mne.io.read_raw_edf(f_path, preload= preload,verbose='WARNING')

def find_artefacts_1d(mask, a_th=0):
    #controllo dei tipi in input
    check_type("mask", mask, [np.ndarray])
    check_type("area_threshold", a_th, [int])

    #controllo delle forme degli input
    if mask.dtype != bool:
        raise TypeError("The data type of `mask` must be numpy.bool, but got %s instead." % mask.dtype)
    if mask.ndim != 1:
        raise ValueError("`mask` must be 1d-numpy.ndarray, "
                         "but got {dimension}d instead.".format(dimension=mask.ndim))
    if a_th < 0:
        raise ValueError("area_threshold must be integer no less than 0, but got %s" % a_th)
    
    res = []
    #prendo tutioni contigui true, cioe gli intervalli accettati
    for k, g in it.groupby(enumerate(mask), key=lambda x: x[1]):
        #k=flag dell'intervallo, g=iteratore sui campioni dell'intervallo
        #se l'intervallo è vero, prendo gli indici 
        if k:
            index, _ = list(zip(*g))
            #se la lunghezza dell'intervallo supera la soglia
            if index[-1] - index[0] >= a_th:
                #creo l'intervallo se la lunghezza supera la soglia
                res.append(P.closed(index[0], index[-1]))
        #restituisco un set di intervalli      
    return res


def find_artefacts_2d(mask, a_th=0):
    #controllo dei tipi in input
    check_type('mask', mask, [np.ndarray])
    check_type('a_th', a_th, [int])

    #controllo delle forme degli input
    if mask.dtype != bool:
        raise TypeError("The data type of `mask` must be numpy.bool, but got %s instead." % mask.dtype)
    if mask.ndim != 2:
        raise ValueError("`mask` must be 2d-numpy.ndarray, "
                         "but got {dimension}d instead.".format(dimension=mask.ndim))
    if a_th < 0:
        raise ValueError("area_threshold must be integer no less than 0, but got %s" % a_th)
    
    #applico la ricerca riga per riga
    func = partial(find_artefacts_1d, a_th=a_th)
    res = list(map(func, mask))
    return res

def interval_set(lista_intervalli):
    """Unisce una lista di intervalli in un singolo oggetto portion.Interval."""
    set_intervalli = P.empty() 
    for intervallo in lista_intervalli:
        set_intervalli = set_intervalli | intervallo
    return set_intervalli

def merge_continuous_artifacts(x, th):
    #controllo i tipi degli input
    check_type('x',x,[list])
    check_type('th',th,[int,float])

    #controllo la dimensione della treashold
    if th <= 0.0:
        raise ValueError("threshold must be float or int > 0.0, but got %s instead" % th)
    #se ho 0 o 1 intervallo lo ritorno direttamente
    n = len(x)
    if n <= 1:
        return interval_set(x)
    #ordino gli intervalli per il lower bound
    x.sort(key=lambda interval: interval.lower)
    #unisco gli intervalli più vicini di una preashold
    res = []
    pre = x[0]
    for i in range(1, n):
        cur = x[i]
        if cur.lower - pre.upper <= th:
            pre = pre | cur
        else:
            res.append(pre)
            pre = cur
    res.append(pre)
    return interval_set(res)
    
def timer(func):
    @wraps(func)
    def wrapper(*arg, **kw):
        start = time.time()
        r = func(*arg, **kw)
        end = time.time()
        total_time = end - start
        print("[%s] Elapsed time: %.5f s." % (func.__name__, total_time))
        return r

    return wrapper

def type_assert(*type_args, **type_kwargs):

    def decorator(func):
        sig = signature(func)
        bound_types = sig.bind_partial(*type_args, **type_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_values = sig.bind(*args, **kwargs)
            for name, value in bound_values.arguments.items():
                if name in bound_types:
                    if not isinstance(value, bound_types[name]):
                        raise TypeError('Argument {} must be {}'.format(name, bound_types[name]))
            return func(*args, **kwargs)

        return wrapper

    return decorator