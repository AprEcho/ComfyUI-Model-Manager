import os


from . import config
from . import utils


class ApiKey:

    def __init__(self):
        self.__cache_file = os.path.join(config.extension_uri, "private.key")
        self.__store: dict[str, str] = {}
        self._load_from_storage()

    def _load_from_storage(self):
        try:
            if os.path.exists(self.__cache_file):
                self.__store = utils.load_dict_pickle_file(self.__cache_file)
        except Exception as e:
            utils.print_error(f"Failed to load api keys from storage: {e}")

    def init(self, request):
        # Try to migrate api key from user setting
        if not os.path.exists(self.__cache_file):
            self.__store = {
                "civitai": utils.get_setting_value(request, "api_key.civitai"),
                "huggingface": utils.get_setting_value(request, "api_key.huggingface"),
            }
            self.__update__()
            # Remove api key from user setting
            utils.set_setting_value(request, "api_key.civitai", None)
            utils.set_setting_value(request, "api_key.huggingface", None)
        else:
            self._load_from_storage()

        # Desensitization returns
        result: dict[str, str] = {}
        for key in self.__store:
            value = self.__store[key]
            if value is not None:
                result[key] = value[:4] + "****" + value[-4:]
        return result

    def get_value(self, key: str):
        self._load_from_storage()
        return self.__store.get(key, None)

    def set_value(self, key: str, value: str):
        self.__store[key] = value
        self.__update__()

    def __update__(self):
        utils.save_dict_pickle_file(self.__cache_file, self.__store)
