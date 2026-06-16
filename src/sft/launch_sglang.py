import os
import sys
from sglang.test.test_utils import is_in_ci

if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd

from sglang.utils import wait_for_server, print_highlight, terminate_process


def add_no_proxy_host(*hosts):
    values = [value for value in os.environ.get("NO_PROXY", "").split(",") if value]
    for host in hosts:
        if host not in values:
            values.append(host)
    no_proxy = ",".join(values)
    os.environ["NO_PROXY"] = no_proxy
    os.environ["no_proxy"] = no_proxy


model_path = sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]

tp = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
server_process, port = launch_server_cmd(f"python3 -m sglang.launch_server --model-path {model_path} --host 0.0.0.0 --mem-fraction-static 0.8 --tp {tp}")

add_no_proxy_host("127.0.0.1", "localhost")
wait_for_server(f"http://localhost:{port}")
print(f"Server started on http://localhost:{port}")
