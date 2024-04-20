import json
import sys
import h5py
import os


def load_config():
    with open('../config.json', 'r') as f:
        config = json.load(f)
    return config


def get_database_path():
    config = load_config()
    if sys.platform.startswith('linux'):
        return config['linux']['database_path']
    elif sys.platform.startswith('win32') or sys.platform.startswith('cygwin'):
        return config['windows']['database_path']
    else:
        raise Exception("Unsupported OS")


def get_h5_files():
    config = load_config()
    return config['bkg_files'], config['sig_files']


def read_h5_file(path, file_name):
    return h5py.File(os.path.join(path, file_name), 'r')
