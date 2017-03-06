from importlib import reload
from . import _settings

def get(key, *args, **kwargs):
    _settings = reload(globals()['_settings'])
    return getattr(_settings, key, *args, **kwargs)
