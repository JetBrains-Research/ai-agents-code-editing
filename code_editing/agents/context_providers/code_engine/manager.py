import socket
import subprocess
import time

import code_engine_client

from code_editing.agents.context_providers.context_provider import ContextProvider


def get_free_port():
    sock = socket.socket()
    sock.bind(('', 0))
    return sock.getsockname()[1]


class CodeEngineManager(ContextProvider):
    def __init__(self, binary_path: str, repo_path: str, **kwargs):
        self.binary_path = binary_path
        self.repo_path = repo_path
        self.server_process = None
        self.api_client = None
        self.start_server()
        self.set_working_dir(repo_path)

    def start_server(self):
        port = get_free_port()
        command = [self.binary_path, f"-port={port}"]
        self.server_process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        server_url = f"http://localhost:{port}"
        self.api_client = code_engine_client.ApiClient(configuration=code_engine_client.Configuration(host=server_url))

        # Wait for server to start
        time.sleep(5)

    def set_working_dir(self, working_dir: str):
        fs_api = code_engine_client.FileSystemApiApi(self.api_client)
        set_working_dir_request = code_engine_client.FileSystemApiSetWorkingDirRequest(working_dir=working_dir)
        fs_api.file_system_set_working_dir_post(set_working_dir_request)

    def __del__(self):
        self.shutdown_server()

    def shutdown_server(self):
        if self.server_process:
            self.server_process.terminate()
            self.server_process = None
