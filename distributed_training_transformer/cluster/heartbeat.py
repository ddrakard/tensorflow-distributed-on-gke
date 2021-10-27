import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import typing
from urllib.error import URLError
from urllib.request import urlopen


heartbeat_message = 'Server operational'


def wait_for_cluster(
        hosts: typing.List, port: int, verbose: bool = False) -> None:
    """
        Start a heartbeat server and wait until all the given hosts have
        started theirs.

        :param hosts: The list of hosts to check for heartbeat servers.
        :param port: The port that all of the hosts are starting their
            servers on.
        :param verbose: Set to True to output information about the progress.
    """
    if verbose:
        print('Starting heartbeat server.')
    start_heartbeat_server(port)
    if verbose:
        print('Trying to connect to peers.')
    for hostname in hosts:
        while True:
            try:
                connection = urlopen('http://' + hostname + ':' + str(port))
                received_message = connection.read().decode('utf-8')
                if received_message != heartbeat_message:
                    raise Exception(
                        'Unexpected response from peer heartbeat server:\n'
                        + received_message)
                break
            except URLError as exception:
                time.sleep(2)
    if verbose:
        print('All peers reachable.')


def start_heartbeat_server(port: int) -> None:
    """
        Start a very simple server to allow other hosts to see if they can
        make a connection.

        :param port: The local port to start the server on.
    """

    class Handler(BaseHTTPRequestHandler):

        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/plain; charset=utf-8')
            self.end_headers()
            self.wfile.write(bytes(heartbeat_message, 'utf-8'))

        def log_request(code='-', size='-'):
            pass

    daemon = threading.Thread(
        name='liveness_server',
        target=lambda: HTTPServer(('', port), Handler).serve_forever())
    # Set as a daemon so it will be killed once the main thread is dead.
    daemon.setDaemon(True)
    daemon.start()
