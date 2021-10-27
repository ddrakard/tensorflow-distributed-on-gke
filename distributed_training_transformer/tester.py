import tensorflow as tf


class Tester(tf.Module):
    """
        A harness to test trained Transformer weights.
    """

    def __init__(self, tokenizers, transformer):
        self.tokenizers = tokenizers
        self.transformer = transformer

    def __call__(self, sentence, max_length=20):
        # input sentence is portuguese, hence adding the start and end token
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]
        sentence = self.tokenizers.pt.tokenize(sentence).to_tensor()
        encoder_input = sentence
        # as the target is english, the first token to the transformer should
        #  be the english start token.
        start_end = self.tokenizers.en.tokenize([''])[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]
        # `tf.TensorArray` is required here (instead of a python list) so that
        #  the dynamic-loop can be traced by `tf.function`.
        output_array = tf.TensorArray(
            dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)
        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions, _ = self.transformer(
                [encoder_input, output], training=False)
            # select the last token from the seq_len dimension
            # (batch_size, 1, vocab_size)
            predictions = predictions[:, -1:, :]
            predicted_id = tf.argmax(predictions, axis=-1)
            # concatentate the predicted_id to the output which is given to the
            # decoder as its input.
            output_array = output_array.write(i+1, predicted_id[0])
            if predicted_id == end:
                break
        # output.shape (1, tokens)
        output = tf.transpose(output_array.stack())
        # shape: ()
        text = self.tokenizers.en.detokenize(output)[0]
        tokens = self.tokenizers.en.lookup(output)[0]
        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop. So recalculate them
        # outside the loop.
        _, attention_weights = self.transformer(
            [encoder_input, output[:,:-1]], training=False)
        return text, tokens, attention_weights
