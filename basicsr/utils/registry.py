# registry.py
import logging

# Setting up a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING) 
handler = logging.FileHandler('registry_warnings.log')
handler.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class Registry:
    """
    A registry to provide name -> object mapping, supporting third-party user custom modules.
    """

    def __init__(self, name):
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj):
        if name in self._obj_map:
            warning_message = f"Warning: An object named '{name}' is already registered in '{self._name}' registry! Skipping duplicate registration."
            print(warning_message)
            logger.warning(warning_message)
            return
        self._obj_map[name] = obj

    def register(self, obj=None):
        if obj is None:
            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class
            return deco
    
        name = obj.__name__
        self._do_register(name, obj)

    def get(self, name):
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        return ret

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()


DATASET_REGISTRY = Registry('dataset')
ARCH_REGISTRY = Registry('arch')
MODEL_REGISTRY = Registry('model')
LOSS_REGISTRY = Registry('loss')
METRIC_REGISTRY = Registry('metric')