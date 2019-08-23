import abc


class Controller(abc.ABCMeta):
    @abc.abstractmethod
    def update(cls, *args):
        raise NotImplementedError

    def get_metric(cls, reset):
        pass

    def predict(cls, *args):
        raise NotImplementedError
