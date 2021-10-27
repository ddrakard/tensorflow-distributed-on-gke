# Running the training process

This project runs the included training code on multiple machines in a
Google Kubernetes cluster. As training progresses, snapshots of the trained
model are uploaded to Google Cloud Storage

⚠️ Warning: this project makes use of potentially expensive cloud resources.
It does not automatically delete the resources after completion. Make sure
you delete resources when you no longer need them, and that you understand
the cost implicatons.

### Preparation

You must have the following things before running the project:

- A Google Cloud Platform account with relevant services enabled.
- A Kubernetes cluster running in GCP.
- Suitable Node Pools configured to run the training nodes.
  These should be configured with GPUs for GPU accelerated training.
- A computer with Python 3, TensorFlow 2, and Docker correctly configured.

Authenticate kubectl on your computer, a guide is available at:
https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl

🛈 We recommend installing [Lens](https://k8slens.dev/) to monitor your
cluster. It can connect using the kubeconfig generated when you authenticated
kubectl.

According to the documentation at
https://cloud.google.com/kubernetes-engine/docs/how-to/gpus the following
command should be run to enable GPU drivers on a GKE cluster:

```
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

Enable the view role for containers in the cluster, so that the workers will
be able to automatically organise themselves. NOTE this performs a cluster-wide
configuration change that affects security. Do this a different way if you are
using an existing cluster (not just for testing). Run the following command
from the root directory of the project:

```
kubectl apply -f kubernetes/enable-view-role.yaml
```

### Codebase configuration

You must configure the codebase so it can run in your Google Cloud account.
Modify the file `configuration/settings.yaml`. Set the
`cloud_storage_bucket_name` field to a Google Cloud Storage bucket that you
have created (make one if you need to) and set the field
`cloud_storage_upload_folder` to a location in that bucket where you want
the training snapshots (output model from training) stored.

You may change the worker count if you want to have more compute power. If you
do so, also change the value of the `replicas:` field in
`kubernetes/stateful-set.yaml` to match it. Make sure you have a large enough
Node Pool.

The codebase needs permissions to access your Google Cloud account. This
project uses a service account access key. Create a service account key
using the Google Cloud web user interface and place it at
`configuration/gcp-access-key.json` in this project (rename the key to match).
This is an easy method to use but may not give appropriate security for
production usage. If you wish to use this project as a basis for production it
is recommended that you study Google Cloud authentication options fully.

### Upload a container image

Build a docker image from this project. This image must be uploaded to
your Artifact Registry in Google Cloud, so it must be tagged correctly. Go to
https://console.cloud.google.com/artifacts and click on the registry you wish
to use (make one if you do not have one already). When the page from the
registry opens, near the top there should be a breadcrumb like the following:

```
us-docker.pkg.dev > my-project > my-repository
```

This tells you the name of the location, Google Cloud project, and repository
you are using.
You will need to include these in the tag for your docker image, as well as a
name for the image. The full tag is:
`location/project-name/repository-name/image-name`, for example
`us-docker.pkg.dev/my-project/my-repository/train-transformer`.

You can now build the image by running the command
`docker build -t {{my-tag}} .` from the root directory of this project, where
{{my-tag}} is the image tag. For example:

```
docker build -t us-docker.pkg.dev/my-project/my-repository/train-transformer .
```

Update the file `kubernetes/stateful-set.yaml` so that the `image:` field
contains your docker image tag.

Now upload the image to your repository by using the command
`docker push {{my-tag}}`, where {{my-tag}} is the same tag, for example

```
docker push us-docker.pkg.dev/my-project/my-repository/train-transformer
```

### Executing the training session

To start the training session, run
`kubectl apply -f kubernites/stateful-set.yaml` from the root directory
of this project. This will create a Stateful Set in your cluster which
will cause worker pods to be created.

Progress is reported via Docker
output logging. You can view the logs by clicking the `View Logs` link
in the Cloud Logging row under the Features heading on the DETAILS tab
of the cluster page in the web interface of Google Cloud Platform. When
doing this you must change the "query" in the panel near the top of the page,
change `resource.type="k8s_cluster"` to `resource.type="k8s_container"`.

The training procedure can be stopped by deleting the Stateful Set (its name
if you have not changed it is `tensorflow-training`). This will stop the pods
automatically. Make sure you do this when training is complete. The Stateful
Set will not automatically terminate itself, so by default it will run
indefinitely, and will keep extra nodes in use if you have auto-scaling
enabled, which can result in unlimited Google Cloud costs. If you do not
have auto-scaling, you additionally need to remove any unused nodes.

### Trying out the model

Trained weights can be tested using the
`distributed_training_transformer/test.py` script.
