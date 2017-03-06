import os.path
import pickle
import numpy as np

settings = None


def config(obj):
    global settings
    settings = obj


def savepickle(filename, obj):
    with open(
            os.path.join(
                str(settings.get('DATA_DIR_INTERCHANGE')),
                "{}.pkl".format(filename)
            ),
            'wb'
            ) as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def loadpickle(filename):
    with open(
        os.path.join(
            str(settings.get('DATA_DIR_INTERCHANGE')),
            "{}.pkl".format(filename)
        ),
        'rb'
    ) as input:
        obj = pickle.load(input)
    return obj


def savenpz(filename, *args, **kwargs):
    return np.savez_compressed(
        os.path.join(
            str(settings.get('DATA_DIR_INTERCHANGE')),
            "{}.npz".format(filename)
        ),
        *args,
        **kwargs
    )


def loadnpz(filename):
    return np.load(
        os.path.join(
            str(settings.get('DATA_DIR_INTERCHANGE')),
            "{}.npz".format(filename)
        )
    )
