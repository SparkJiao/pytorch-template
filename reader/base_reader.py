import abc


class BaseReader(abc.ABCMeta):
    def read(cls, *args, **kwargs):
        raise NotImplementedError
