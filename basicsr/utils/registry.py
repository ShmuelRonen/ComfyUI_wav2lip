# registry.py
# Modified from: https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/registry.py  # noqa: E501
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
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.
    """

    def __init__(self, name):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj):
        if name in self._obj_map:
            warning_message = (f"Warning: An object named '{name}' is already registered in "
                               f"'{self._name}' registry! Skipping duplicate registration.")
            print(warning_message)
            logger.warning(warning_message)
            return
        self._obj_map[name] = obj

    def register(self, obj=None):
        """
        Register the given object under the name `obj.__name__`.
        Can be used as either a decorator or not.
        See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__
        self._do_register(name, obj)

    def get(self, name):
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        return ret

    def __contains__(

