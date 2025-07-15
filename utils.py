import os
import mne

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
        for name in files:
            files = os.listdir(d_path)
            if accept_file(d_path, name, f_extensions):
                path_list.append(os.path.join(d_path, name))
    
    return path_list

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
    return mne.io.read_raw_edf(f_path, preload,verbose='WARNING')


