



# Convert the dict to a object.


class Config(dict):
    '''
    >>> cfg = Config({1:2}, a=3)
    Traceback (most recent call last):
    ...
    TypeError: attribute name must be string, not 'int'
    >>> cfg = Config(a=1, b=2)
    >>> cfg.a
    1
    >>> cfg['a']
    1
    >>> cfg['c'] = 3
    >>> cfg.c
    3
    >>> cfg.d = 4
    >>> cfg['d']
    Traceback (most recent call last):
    ...
    KeyError: 'd'
    '''
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        for name, attr in self.items():
            self.__setattr__(name, attr)

    def __setitem__(self, key, value):
        super(Config, self).__setitem__(key, value)
        self.__setattr__(key, value)





if __name__ == "__main__":
    import doctest
    doctest.testmod()











