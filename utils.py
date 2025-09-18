import os
import mne
import time
import numpy as np
from functools import partial
import itertools as it
import portion as P
from functools import wraps
from inspect import signature
import re

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

def get_raw(f_path, start=None, end=None,preload=False):
    check_type('f_path', f_path, [str])
    if os.path.isdir(f_path):
        raise KeyError('can`t identify %s.'%f_path)
    return mne.io.read_raw_edf(f_path, preload=preload,verbose=False)

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

def stride_data(x, n_per_seg, n_overlap):
    if n_per_seg == 1 and n_overlap == 0:
        result = x[..., np.newaxis]
    else:
        step = n_per_seg - n_overlap
        shape = x.shape[:-1]+((x.shape[-1]-n_overlap)//step, n_per_seg)
        strides = x.strides[:-1]+(step*x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return result

def check_channel_names(raw_obj, verbose):
        ch_map = ["EEG FP1-REF", "EEG FP1-LE", "EEG FP1", "FP1",
        "EEG FP2-REF", "EEG FP2-LE", "EEG FP2", "FP2",
        "EEG F3-REF", "EEG F3-LE", "EEG F3", "F3",
        "EEG F4-REF", "EEG F4-LE", "EEG F4", "F4",
        "EEG FZ-REF", "EEG FZ-LE", "EEG FZ", "FZ",
        "EEG F7-REF", "EEG F7-LE", "EEG F7", "F7",
        "EEG F8-REF", "EEG F8-LE", "EEG F8", "F8",
        "EEG P3-REF", "EEG P3-LE", "EEG P3", "P3",
        "EEG P4-REF", "EEG P4-LE", "EEG P4", "P4",
        "EEG PZ-REF", "EEG PZ-LE", "EEG PZ", "PZ",
        "EEG C3-REF", "EEG C3-LE", "EEG C3", "C3",
        "EEG C4-REF", "EEG C4-LE", "EEG C4", "C4",
        "EEG CZ-REF", "EEG CZ-LE", "EEG CZ", "CZ",
        "EEG T3-REF", "EEG T3-LE", "EEG T3", "T3",
        "EEG T4-REF", "EEG T4-LE", "EEG T4", "T4",
        "EEG T5-REF", "EEG T5-LE", "EEG T5", "T5",
        "EEG T6-REF", "EEG T6-LE", "EEG T6", "T6",
        "EEG O1-REF", "EEG O1-LE", "EEG O1", "O1",
        "EEG O2-REF", "EEG O2-LE", "EEG O2", "O2"]
        ch_necessary = ['FP1', 'FP2', 'F3', 'F4', 'FZ', 'F7', 'F8', 'P3', 'P4', 'PZ', 'C3', 'C4', 'CZ', 'T3', 'T4', 'T5', 'T6', 'O1', 'O2']
        #mappa dei nomi dai possibili usati ad uno standard semplice
        ch_mapper = {}
        for i in range(0, len(ch_map), 4):
            standard_name = ch_map[i + 3]
            for j in range(4):
                ch_mapper[ch_map[i + j]] = standard_name
        
        #rinomino i canali
        existing_channels = set(raw_obj.info['ch_names'])
        filtered_ch_mapper = {old: new for old, new in ch_mapper.items() if old in existing_channels}
        raw_obj.rename_channels(filtered_ch_mapper)
        ch_names = set(raw_obj.ch_names)

        #controllo che tutti i nomi dei canali siano presenti
        if set(ch_necessary).issubset(ch_names):
            raw_obj.pick(ch_necessary)
        else:
            raise RuntimeError("Channel Error")
        
def parse_summary_txt(txt_path):

    data = []
    current = {}
    sampling_rate = None
    channels = []

    seizure_start_pattern = re.compile(r'Seizure(?: \d+)? Start Time: (\d+) seconds')
    seizure_end_pattern = re.compile(r'Seizure(?: \d+)? End Time: (\d+) seconds')

    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Frequenza di campionamento
            if line.startswith("Data Sampling Rate:"):
                sampling_rate = float(line.split(":",1)[1].strip().split()[0])

            # Lista canali
            elif line.startswith("Channels in EDF Files:"):
                channels = []
                continue

            elif line.startswith("Channel "):
                label = line.split(":",1)[1].strip()
                channels.append(label)

            # Inizio di un nuovo file
            elif line.startswith("File Name:"):
                if 'File Name' in current:
                    current.setdefault("Seizure Start Times", [])
                    current.setdefault("Seizure End Times", [])
                    if current.get("Seizure Count", 0) > 0 and not current["Seizure Start Times"]:
                        print(f"Warning: {current['File Name']} ha Seizure Count > 0 ma nessun inizio/fine segnalato")
                    data.append(current.copy())
                current = {"File Name": line.split(":",1)[1].strip()}

            elif line.startswith("File Start Time:"):
                current["Start Time"] = line.split(":",1)[1].strip()

            elif line.startswith("File End Time:"):
                current["End Time"] = line.split(":",1)[1].strip()

            elif line.startswith("Number of Seizures in File:"):
                current["Seizure Count"] = int(line.split(":",1)[1].strip())

            else:
                # Cattura tutti i Seizure Start/End numerati o non numerati
                m_start = seizure_start_pattern.match(line)
                m_end = seizure_end_pattern.match(line)
                if m_start:
                    current.setdefault("Seizure Start Times", []).append(int(m_start.group(1)))
                elif m_end:
                    current.setdefault("Seizure End Times", []).append(int(m_end.group(1)))

        # aggiungo l'ultimo file
        if current:
            current.setdefault("Seizure Start Times", [])
            current.setdefault("Seizure End Times", [])
            if current.get("Seizure Count",0) > 0 and not current["Seizure Start Times"]:
                print(f"Warning: {current['File Name']} ha Seizure Count > 0 ma nessun inizio/fine segnalato")
            data.append(current)

    # creo il DataFrame
    df = pd.DataFrame(data)

    # metadati generali
    metadata = {
        "Sampling Rate": sampling_rate,
        "Channels": channels
    }

    return metadata, df

def to_monopolar(raw_bipolar, ref='average', ch_name_sep='-'):
    ch_names = raw_bipolar.ch_names
    data_bip, times = raw_bipolar.get_data(return_times=True)
    data_bip = data_bip.astype(np.float32)
    n_bip, n_samples = data_bip.shape

    pairs = []
    for ch in ch_names:
        if ch_name_sep in ch:
            a, c = ch.split(ch_name_sep, 1)
            a, c = a.strip().upper(), c.strip().upper()
            pairs.append((a, c))
        else:
            raise ValueError(f"Channel name '{ch}' non contiene il separatore '{ch_name_sep}'")

    electrodes = sorted(set([a for a, _ in pairs] + [c for _, c in pairs]))
    n_ele = len(electrodes)

    A = np.zeros((n_bip, n_ele), dtype=float)
    for i, (a, c) in enumerate(pairs):
        A[i, electrodes.index(a)] =  1.0
        A[i, electrodes.index(c)] = -1.0

    A_pinv = np.linalg.pinv(A)
    V = A_pinv.dot(data_bip)

    if isinstance(ref, list):
        ref_found = None
        for r in ref:
            r = r.upper()
            if r in electrodes:
                ref_found = r
                break
        if ref_found is None:
            raise ValueError(f"Nessun riferimento trovato in {ref}, disponibili: {electrodes}")
        ref = ref_found  

    if ref == 'average':
        V = V - V.mean(axis=0, keepdims=True)
    elif isinstance(ref, str):
        ref = ref.upper()
        if ref not in electrodes:
            raise ValueError(f"Riferimento {ref} non trovato negli elettrodi {electrodes}")
        ref_idx = electrodes.index(ref)
        V = V - V[ref_idx:ref_idx+1, :]
    elif ref is None:
        pass
    else:
        raise ValueError("Param ref deve essere 'average', None, stringa o lista di stringhe.")

    sfreq = raw_bipolar.info['sfreq']
    info = mne.create_info(ch_names=electrodes, sfreq=sfreq, ch_types='eeg')
    raw_monopolar = mne.io.RawArray(V, info, verbose=False)

    reconstructed_bip = A.dot(V)
    max_diff = np.max(np.abs(reconstructed_bip - data_bip))

    return raw_monopolar, max_diff

def select_bipolar(raw):
    ch_necessary = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'CZ']  
    flag = False

    selected_chs = []
    for ch in raw.ch_names:
        if '-' in ch:  # canale bipolare
            a, c = ch.upper().split('-', 1)
            if a in ch_necessary or c in ch_necessary:
                selected_chs.append(ch)
                flag = True
        else:  # canale già monopolare
            if ch.upper() in ch_necessary:
                selected_chs.append(ch)
    return raw.pick(selected_chs), flag

def strip_eeg_prefix(raw):
    mapping = {ch: ch.replace('EEG ', '') for ch in raw.info['ch_names'] if ch.startswith('EEG ')}
    raw.rename_channels(mapping)
