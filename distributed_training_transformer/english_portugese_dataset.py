import tensorflow as tf
import tensorflow_datasets as tfds


BUFFER_SIZE = 20000


def tokenizers():
    if not hasattr(tokenizers, 'result'):
        model_name = "ted_hrlr_translate_pt_en_converter"
        tf.keras.utils.get_file(
            model_name + ".zip",
            "https://storage.googleapis.com/download.tensorflow.org/models/"
            + model_name + ".zip",
            cache_dir='.', cache_subdir='temporary', extract=True
        )
        tokenizers.result = tf.saved_model.load('temporary/' + model_name)
    return tokenizers.result


def dataset_batches(global_batch_size: int):
    splits, metadata = tfds.load(
        'ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
    training_set = splits['train']
    validation_set = splits['validation']

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA)
    training_set = training_set.with_options(options)
    validation_set = validation_set.with_options(options)


    def tokenize(portuguese, english):
        return (
            tokenizers().pt.tokenize(portuguese).to_tensor(),
            tokenizers().en.tokenize(english).to_tensor()
        )

    def prepare_dataset(dataset):
        return (
            dataset
            .cache()
            .shuffle(BUFFER_SIZE)
            .batch(global_batch_size)
            .map(tokenize, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))

    return {
        'train': prepare_dataset(training_set),
        'validation': prepare_dataset(validation_set)}
