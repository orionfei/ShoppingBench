"""Small ujson compatibility shim for environments without the binary wheel."""

from json import JSONDecodeError, dump, load, loads
from json import dumps as _json_dumps

__version__ = "stdlib-json-shim"


def dumps(obj, *args, **kwargs):
    kwargs.pop("escape_forward_slashes", None)
    kwargs.pop("encode_html_chars", None)
    kwargs.pop("reject_bytes", None)
    return _json_dumps(obj, *args, **kwargs)
