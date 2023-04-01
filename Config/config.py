from os.path import dirname, join
from Utils import read_json


class OnlyReadConfig:
    attrs = {}

    def _unable_write(self):
        self.enable_write = False

    def __setattr__(self, key, value):

        if not self.enable_write:
            self.__dict__[key] = value
        else:
            raise ValueError("该类为只读类，不可写。")

    def _setattr(self, attrs):
        for key, value in list(attrs.items()):
            if isinstance(value, dict):
                setattr(self, key, AttrConfig(value))
            else:
                setattr(self, key, value)

    def __getattr__(self, key):
        return None

    def __str__(self):
        return str(self.attrs)

    def dict(self):
        return self.attrs


class AttrConfig(OnlyReadConfig):
    def __init__(self, attrs=None):
        self.attrs = attrs
        self._setattr(attrs)
        self._unable_write()


class Config(OnlyReadConfig):

    def __new__(cls):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls)
            cls._flag = True
        return cls._instance

    def __init__(self):
        if self._flag:
            path = join(dirname(__file__), "config.json")
            self.attrs = read_json(path)
            self._setattr(self.attrs)
            self._flag = False
            self._unable_write()
