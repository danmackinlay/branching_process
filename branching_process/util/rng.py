from numpy.random import get_state, set_state, seed


class Seed(object):
    """
    context manager for reproducible seeding.

    >>> with Seed(5):
    >>>     print(np.random.rand())

    0.22199317108973948
    """
    def __init__(self, seed):
        self.seed = seed
        self.state = None

    def __enter__(self):
        self.state = get_state()
        seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        set_state(self.state)
