import json
import pickle
import pandas as pd

CONF_DEFAULT = "config/config_default.json"
BASE_CONF = "config.json"
TMP_DIR = "tmp/"

def load_results(path):
    """Loads the results pickle to an object"""
    with open(path, "rb") as res_input:
        try:
            return pickle.load(res_input)
        except UnicodeDecodeError as e:
            print("Pickle file created using Python 2")
            #print(e)
            print("Trying to reload")
            return pickle.load(res_input,fix_imports=True,encoding='bytes')


def save_results(path, results):
    """Saves the results object to a pickle"""
    with open(path, "wb") as res_output:
        pickle.dump(results, res_output)


def load_ratings(path):
    """Loads ratings from a file to a Pandas DataFrame"""
    return pd.read_csv(path, sep="\t", names=("user_id", "item_id", "rating"))


def recursive_merge(dest, src):
    for k, v in src.items():
        if k in dest and isinstance(v, dict) and isinstance(dest[k], dict):
            recursive_merge(dest[k], v)
        else:
            dest[k] = v


def load_configs(*addresses):
    """Loads multiple configuration files in order, overwriting attributes."""
    conf = {}
    for addr in addresses:
        if addr:
            with open(addr, "r") as f:
                recursive_merge(conf, json.load(f))
    return conf

