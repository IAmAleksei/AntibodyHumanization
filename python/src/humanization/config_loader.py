import os

from configloader import ConfigLoader

config_abs_path = "/".join(os.path.abspath(__file__).split('/')[:-1])


class MetaSingleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(MetaSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Config(metaclass=MetaSingleton):
    def __init__(self, config_file_path=f'{config_abs_path}/config.yaml'):
        self.config_loader = ConfigLoader()
        self.config_loader.update_from_yaml_file(config_file_path)

    def get(self, setting_name):
        return self.config_loader.get(setting_name, None)

    def __getitem__(self, item):
        return self.get(item)

    def to_dict(self):
        loader = self.config_loader
        return {key: loader.get(key) for key in loader.keys()}


LOGGING_LEVEL = "LOGGING_LEVEL"
LOGGING_FORMAT = "LOGGING_FORMAT"

MAX_CHANGES = "MAX_CHANGES"
ANARCI_NCPU = "ANARCI_NCPU"
