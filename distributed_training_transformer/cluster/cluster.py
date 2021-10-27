import contextlib
import json
import os
import time

import tensorflow as tf
from kubernetes import client, config

from distributed_training_transformer.cluster import heartbeat


class TensorflowKubernetesCluster:

    def __init__(
            self, worker_count: int, kubernetes_namespace: str = 'default',
            heartbeat_port: int = 3479, tensorflow_port: int = 3480,
            verbose: bool = False):
        """
            Set up a Tensorflow distributed training session from a node of
            a Stateful Set in a Kubernetes cluster.

            :param worker_count: How many workers (nodes in the Stateful Set)
                there are.
            :param kubernetes_namespace: The kubernetes namespace everything
                is running in.
            :param heartbeat_port: A port to use for auto-managing the training
                session. It should be any free port number.
            :param tensorflow_port: A port to use for the distributed training
                communication. It should be any free port number.
            :param verbose: Set to True to have messages about the progress
                in setting up and managing the cluster are desired.
        """
        self._tensorflow_port = tensorflow_port
        self._verbose = verbose
        if 'THIS_POD_NAME' not in os.environ:
            print(
                'Environment variable THIS_POD_NAME not set, running in local'
                + ' mode.')
            self._ip_addresses = ['127.0.0.1']
            self._pod_number = 0
        else:
            pod_name = os.environ['THIS_POD_NAME']
            self._pod_name_base = pod_name.rsplit('-')[0]
            self._pod_number = int(pod_name.rsplit('-', maxsplit=1)[1])
            if verbose:
                print("Pod name from environment variable: " + pod_name)
            self._set_ip_addresses_from_kubernetes(
                worker_count, kubernetes_namespace)
            heartbeat.wait_for_cluster(
                self._ip_addresses, heartbeat_port, verbose=verbose)
            if self._pod_number != 0:
                # This container is not the chief, so give the chief a little
                #  time to get ready and open the Tensorflow distributed port.
                #  It would be much better if there was a clean way to do this.
                time.sleep(10)
        tf_config = {
            'cluster': {
                'worker': []
            },
            'task': {'type': 'worker', 'index': self._pod_number}
        }
        for ip_address in self._ip_addresses:
            tf_config['cluster']['worker'].append(
                ip_address + ':' + str(self._tensorflow_port))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)
        self._execution_strategy = tf.distribute.MultiWorkerMirroredStrategy()

    def execution_strategy(self) -> tf.distribute.MultiWorkerMirroredStrategy:
        """
            Get the tf.distribute.MultiWorkerMirroredStrategy object
        """
        return self._execution_strategy

    def chief(self) -> bool:
        """
            Is the local process the "chief" of the job, which has a special
            role in Tensorflow distributed training.
        """
        return self._pod_number == 0

    def _set_ip_addresses_from_kubernetes(
            self, worker_count, namespace) -> None:
        """
            Get the IP addresses of the workers by using the API of the
            enclosing Kubernetes cluster.
        """
        config.load_incluster_config()
        api = client.CoreV1Api()
        if self._verbose:
            print(
                'Waiting until ' + str(worker_count - 1)
                + ' peers become available.')
        while True:
            workers = []
            for pod in api.list_namespaced_pod(namespace).items:
                name_parts = pod.metadata.name.rsplit('-')
                if (
                        name_parts[0] == self._pod_name_base
                        and pod.status.pod_ip is not None):
                    workers.append(pod)
            if len(workers) == worker_count:
                break
            time.sleep(2)
        workers.sort(key=lambda pod: pod.metadata.name)
        if self._verbose:
            print("Found pods in stateful set:")
            for pod in workers:
                print(pod.metadata.name + ' ' + pod.status.pod_ip)
        self._ip_addresses = [pod.status.pod_ip for pod in workers]