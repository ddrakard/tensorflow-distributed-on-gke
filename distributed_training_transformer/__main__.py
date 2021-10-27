"""
    Train a Transformer model from within Kubernetes using Tensorflow 2
    distributed.

    Based on the guide
    https://www.tensorflow.org/text/tutorials/transformer
"""
import time

import yaml
import tensorflow as tf

from distributed_training_transformer.cluster.cluster import (
    TensorflowKubernetesCluster)
from distributed_training_transformer.transformer_model import (
    Transformer, accuracy_function, local_loss_function)
from distributed_training_transformer import checkpoint, \
    english_portugese_dataset, transformer_model

with open('configuration/settings.yaml', 'r') as file:
    settings = yaml.safe_load(file)
WORKER_COUNT = settings['worker_count']
LOCAL_BATCH_SIZE = settings['local_batch_size']
UPLOAD_BUCKET_NAME = settings['cloud_storage_bucket_name']
UPLOAD_FOLDER_BASE_NAME = settings['cloud_storage_upload_folder']


cluster = TensorflowKubernetesCluster(WORKER_COUNT, verbose=True)

GOOGLE_CLOUD_ACCESS_KEY_PATH = 'configuration/gcp-access-key.json'
GLOBAL_BATCH_SIZE = LOCAL_BATCH_SIZE * WORKER_COUNT
INPUT_VOCABULARY_SIZE = (
    english_portugese_dataset.tokenizers().pt.get_vocab_size().numpy())
TARGET_VOCABULARY_SIZE = (
    english_portugese_dataset.tokenizers().en.get_vocab_size().numpy())

with cluster.execution_strategy().scope():

    layers_count = 4
    embedding_size = 128
    feed_forward_hidden_size = 512
    heads_count = 8
    dropout_rate = 0.1

    datasets_batches = english_portugese_dataset.dataset_batches(
        GLOBAL_BATCH_SIZE)

    training_batches = (
        cluster.execution_strategy().experimental_distribute_dataset(
            datasets_batches['train']))
    validation_batches = (
        cluster.execution_strategy().experimental_distribute_dataset(
            datasets_batches['validation']))


    class TransformerLearningRateSchedule(
            tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, embedding_size, warmup_steps=4000):
            super().__init__()
            self.embedding_size = tf.cast(embedding_size, tf.float32)
            self.warmup_steps = warmup_steps

        def __call__(self, step):
            rising_stage = step * (self.warmup_steps ** -1.5)
            falling_stage = tf.math.rsqrt(step)
            return tf.math.rsqrt(self.embedding_size) * tf.math.minimum(
                rising_stage, falling_stage)


    learning_rate = TransformerLearningRateSchedule(embedding_size)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    transformer = Transformer(
        layers_count=layers_count,
        embedding_size=embedding_size,
        heads_count=heads_count,
        feed_forward_hidden_size=feed_forward_hidden_size,
        input_vocabulary_size=INPUT_VOCABULARY_SIZE,
        target_vocabulary_size=TARGET_VOCABULARY_SIZE,
        maximum_input_length=1000,
        maximum_target_length=1000,
        droput_rate=dropout_rate)
    transformer_model.build_model(transformer)

    transformer.load_weights('saved_weights/2/model_weights')

    EPOCHS = 20

    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='train_loss')
    test_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    # The @tf.function trace-compiles train_step into a TF graph for faster
    # execution. The function specializes to the precise shape of the argument
    # tensors. To avoid re-tracing due to the variable sequence lengths or variable
    # batch sizes (the last batch is smaller), use input_signature to specify
    # more generic shapes.
    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    def train_step(input, target):
        target_input = target[:, :-1]
        target_real = target[:, 1:]
        with tf.GradientTape() as tape:
            predictions, _ = transformer(
                [input, target_input], training = True)
            loss = local_loss_function(target_real, predictions, WORKER_COUNT)
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, transformer.trainable_variables))
        train_accuracy.update_state(
            accuracy_function(target_real, predictions))
        return loss

    def test_step(input, target):
        target_input = target[:, :-1]
        target_real = target[:, 1:]
        predictions, _ = transformer([input, target_input], training=False)
        test_loss.update_state(
            local_loss_function(target_real, predictions, 1))
        test_accuracy.update_state(accuracy_function(target_real, predictions))

    @tf.function(experimental_relax_shapes=True)
    def distributed_train_step(input, target):
        per_replica_losses = cluster.execution_strategy().run(
            train_step, args=(input, target))
        return cluster.execution_strategy().reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    @tf.function(experimental_relax_shapes=True)
    def distributed_test_step(input, target):
        per_replica_losses = cluster.execution_strategy().run(
            test_step, args=(input, target))

    if cluster.chief():
        uploader = checkpoint.ModelUploader(
            UPLOAD_BUCKET_NAME, UPLOAD_FOLDER_BASE_NAME,
            GOOGLE_CLOUD_ACCESS_KEY_PATH, 'temporary')
        uploader.take_snapshot(transformer)
        print(
            'Initial model snapshot to '
            + uploader.last_upload_location())
        initial_snapshot_saved = True

    for epoch in range(EPOCHS):
        start = time.time()
        total_train_loss = 0.0
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        # input -> portuguese, target -> english
        for (batch, (input, target)) in enumerate(training_batches):
            # Convert to 1-based for calculations
            total_batches = batch + 1
            total_train_loss += distributed_train_step(input, target)
            if batch % 50 == 0:
                print(
                    f'Epoch {epoch + 1} Batch {batch}'
                    + f' Loss {(total_train_loss / total_batches):.4f}'
                    + f' Accuracy {train_accuracy.result():.4f}')
        if cluster.chief() and (epoch + 1) % 5 == 0:
            uploader.take_snapshot(transformer)
            print(
                f'Saving checkpoint for epoch {epoch + 1}'
                + f' at {uploader.last_upload_location()}')

        for (batch, (input, target)) in enumerate(validation_batches):
            distributed_test_step(input, target)
        print(
            f'Epoch {epoch + 1}'
            + f' Train Loss {(total_train_loss/total_batches):.4f}'
            + f' Train Accuracy {train_accuracy.result():.4f}'
            + f' Test Loss {test_loss.result():.4f}'
            + f' Test Accuracy {test_accuracy.result():.4f}')
        print(
            f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')


print("Training complete, script in idle mode.")

while True:
    time.sleep(1)
