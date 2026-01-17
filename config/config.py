import os
import tempfile
import python_lua_helper


class ConfigLoader:
    def __init__(
        self,
        lua_config_script: str,
        temp_dir: str = None,
    ):
        # prepare temp directory
        self._temp_dir = temp_dir
        if self._temp_dir:
            if not os.path.exists(self._temp_dir):
                raise ValueError(f"Temp directory does not exist: {self._temp_dir}")
            self._temp_dir = tempfile.mkdtemp(prefix="llm-gw-", dir=self._temp_dir)
        else:
            self._temp_dir = tempfile.mkdtemp(prefix="llm-gw-")
        self._temp_dir = os.path.abspath(self._temp_dir)
        # load and parse lua config
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self._cfg = python_lua_helper.PyLuaHelper(
            lua_config_script=lua_config_script,
            export_vars=["server", "models"],
            pre_script=os.path.join(script_dir, "pre.lua"),
            post_script=os.path.join(script_dir, "post.lua"),
            work_dir=script_dir,
        )

    @property
    def cfg(self):
        return self._cfg

    @property
    def temp_dir(self):
        return self._temp_dir
