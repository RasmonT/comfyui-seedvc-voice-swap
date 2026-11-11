# Vendored from https://github.com/Plachtaa/seed-vc (commit 51383ef), GPL-3.0.
# Modified for comfyui-seedvc-voice-swap: only AttrDict kept.
# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.



class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
