# Transformer distributed training on GKE

This project demonstrates the training of a Transformer on GCP GKE.
Distributed training and GPUs are used.

The project uses 4 main technologies to do this:
- TensorFlow 2 on Python 3 for the distributed model training and dynamic
 (during execution) control of the training process, including integrating with
  Kubernetes and Google Cloud.
- Docker to package the code in a standard way so it can be run on an execution
  platform (in this case Kubernetes).
- Kubernetes as a distributed execution platform.
- Google Cloud for cloud file storage and to host the Kubernetes cluster.

The project assumes you have a basic understanding of all the above
technologies, and is intended as a demonstration of how to get them to run
together end-to-end. It is not intended as an individual first introduction to
them.

To get started using the project, see
[documentation/running.md](documentation/running.md)

For any feedback, questions, or contributions, please see
[documentation/contributing.md](documentation/contributing.md)

Developed in association with [ML Collective](https://mlcollective.org/)