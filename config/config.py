import os
import python_lua_helper


class ConfigLoader:
    def __init__(
        self,
        lua_config_script: str,
    ):
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
