import keras
import tensorflow as tf
# Required for saved_model.load to work correctly with the tokenizer
import tensorflow_text


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def local_loss_function(actual, predicted, workers_count):
    mask = tf.math.logical_not(tf.math.equal(actual, 0))
    loss_ = loss_object(actual, predicted)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    local_loss = tf.reduce_sum(loss_) / tf.reduce_sum(mask)
    return local_loss / workers_count


def accuracy_function(actual, predicted):
    accuracies = tf.equal(actual, tf.argmax(predicted, axis=2))
    mask = tf.math.logical_not(tf.math.equal(actual, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


def positional_encoding(sentence_size, embedding_size):

    def cast_float(value):
        return tf.cast(value, tf.float32)

    def get_angles(sentence_position, embedding_position):
        angle_rates = (
            1 / tf.pow(
                10000,
                (
                    cast_float((2 * (embedding_position // 2)))
                    / cast_float(embedding_size))))
        return cast_float(sentence_position) * angle_rates

    angles = get_angles(
        tf.expand_dims(tf.range(sentence_size), -1),
        tf.expand_dims(tf.range(embedding_size), 0))
    # Note: implementation could be made faster with reshape and transpose
    # instead of masking.
    odd_mask = tf.math.mod(tf.expand_dims(tf.range(embedding_size), 0), 2)
    even_mask = 1 - odd_mask
    result = (
        cast_float(odd_mask) * tf.math.cos(angles)
        + cast_float(even_mask) * tf.math.sin(angles))
    return tf.expand_dims(result, 0)


def create_padding_mask(sequence):
    """
        0 indicates mask on, 1 indicates mask off.
    """
    result = tf.cast(tf.math.equal(sequence, 0), tf.float32)
    # add extra dimensions to add the padding to the attention logits.
    return tf.expand_dims(tf.expand_dims(result, 1), 1)


def create_look_ahead_mask(size):
    """
        0 indicates mask on, 1 indicates mask off.
    """
    # 1 - lower triangular matrix
    return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)


def scaled_dot_product_attention(query, keys, values, mask=None):
    """

        query, key, and value must have matching leading dimensions.

        :param query: shape (..., sequence_length, input_mapped_embedding_size)
            "input_mapped_embedding_size" makes the assumption that the query
            and key multiplied axes are derived from a transformation of some
            prior embedding (as is the case in Transformer architecture).
        :param keys: shape (..., values_count, input_mapped_embedding_size)
            Axes are reversed because it will be transposed as part of the
            dot product.
        :param values: shape (..., values_count, output_embedding_size)
        :param mask: shape broadcastable to
            (..., sequence_length, values_count)
        :return: output, attention_weights
    """
    # In the case of self attention, the sequence_length and values_count
    #  will be the same.

    # (..., sequence_length, values_count)
    raw_attention_logits = tf.matmul(query, keys, transpose_b=True)
    key_dimension = tf.cast(tf.shape(keys)[-1], tf.float32)
    scaled_attention_logits = raw_attention_logits / tf.math.sqrt(key_dimension)

    # Add the mask to the scaled tensor. Large negative values go to zero in
    #  the softmax.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (values_count) so that the scores
    # add up to 1.
    # (..., sequence_length, values_count)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    # (..., sequence_length, output_embedding_size)
    output = tf.matmul(attention_weights, values)
    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embedding_size, heads_count):
        super(MultiHeadAttention, self).__init__()
        self.heads_count = heads_count
        self.embedding_size = embedding_size
        assert embedding_size % self.heads_count == 0
        self.head_size = embedding_size // self.heads_count
        self.query_generator_weights = tf.keras.layers.Dense(embedding_size)
        self.key_generator_weights = tf.keras.layers.Dense(embedding_size)
        self.value_generator_weights = tf.keras.layers.Dense(embedding_size)
        self.dense = tf.keras.layers.Dense(embedding_size)

    def split_heads(self, x, batch_size):
        """
            Split the last dimension into (heads_count, depth).
            Transpose the result such that the shape is (batch_size,
            heads_count, input_sequence_length, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.heads_count, self.head_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, values, keys, query, mask):
        batch_size = tf.shape(query)[0]
        # (batch_size, input_sequence_length, embedding_size)
        query = self.query_generator_weights(query)
        # (batch_size, values_count, embedding_size)
        # If the values are derived from the input sequence, as they are in
        #  self-attention, then values_count will be the same as
        #  input_sequence_length.
        keys = self.key_generator_weights(keys)
        # (batch_size, values_count, embedding_size)
        values = self.value_generator_weights(values)

        # (batch_size, heads_count, input_sequence_length, head_size)
        query = self.split_heads(query, batch_size)
        # (batch_size, heads_count, input_sequence_length, head_size)
        keys = self.split_heads(keys, batch_size)
        # (batch_size, heads_count, input_sequence_length, head_size)
        values = self.split_heads(values, batch_size)

        # scaled_attention shape
        #   (batch_size, heads_count, input_sequence_length, head_size)
        # attention_weights shape
        #   (batch_size, heads_count, input_sequence_length,
        #   input_sequence_length)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            query, keys, values, mask)
        # (batch_size, input_sequence_length, heads_count, head_size)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # (batch_size, input_sequence_length, embedding_size)
        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.embedding_size))
        # (batch_size, input_sequence_length, embedding_size)
        output = self.dense(concat_attention)
        return output, attention_weights


def point_wise_feed_forward_network(embedding_size, hidden_size):
    return tf.keras.Sequential([
        # (batch_size, sequence_length, hidden_size)
        tf.keras.layers.Dense(hidden_size, activation='relu'),
        # (batch_size, sequence_length, embedding_size)
        tf.keras.layers.Dense(embedding_size)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(
            self, embedding_size, heads_count, feed_forward_hidden_size,
            dropout_rate=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(embedding_size, heads_count)
        self.ffn = point_wise_feed_forward_network(
            embedding_size, feed_forward_hidden_size)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        # (batch_size, sequence_length, embedding_size)
        attention_output, _ = self.mha(x, x, x, mask)
        attention_output = self.dropout1(attention_output, training=training)
        # (batch_size, sequence_length, embedding_size)
        output1 = self.layernorm1(x + attention_output)

        # (batch_size, sequence_length, embedding_size)
        ffn_output = self.ffn(output1)
        ffn_output = self.dropout2(ffn_output, training=training)
        # (batch_size, sequence_length, embedding_size)
        return self.layernorm2(output1 + ffn_output)


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(
            self, embedding_size, heads_count, feed_forward_hidden_size,
            rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(embedding_size, heads_count)
        self.mha2 = MultiHeadAttention(embedding_size, heads_count)

        self.ffn = point_wise_feed_forward_network(
            embedding_size, feed_forward_hidden_size)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        # enc_output shape (batch_size, sequence_length, embedding_size)

        # (batch_size, sequence_length, embedding_size)
        output1, attention_weights_1 = self.mha1(x, x, x, look_ahead_mask)
        output1 = self.dropout1(output1, training=training)
        output1 = self.layernorm1(output1 + x)

        # (batch_size, sequence_length, embedding_size)
        output2, attention_weights_2 = self.mha2(
            encoder_output, encoder_output, output1, padding_mask)
        output2 = self.dropout2(output2, training=training)
        # (batch_size, sequence_length, embedding_size)
        output2 = self.layernorm2(output2 + output1)

        # (batch_size, sequence_length, embedding_size)
        ffn_output = self.ffn(output2)
        ffn_output = self.dropout3(ffn_output, training=training)
        # (batch_size, sequence_length, embedding_size)
        output3 = self.layernorm3(ffn_output + output2)

        return output3, attention_weights_1, attention_weights_2


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(
            self, layers_count, embedding_size, heads_count,
            feed_forward_hidden_size, input_vocabulary_size,
            maximum_position_encoding, dropout_rate=0.1):
        super().__init__()
        self.embedding_size = embedding_size
        self.layers_count = layers_count
        self.embedding = tf.keras.layers.Embedding(
            input_vocabulary_size, embedding_size)
        self.positional_encoding = positional_encoding(
            maximum_position_encoding, embedding_size)
        self.encoder_layers = [
            EncoderLayer(
                embedding_size, heads_count, feed_forward_hidden_size,
                dropout_rate)
            for _ in range(layers_count)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        sequence_length = tf.shape(x)[1]
        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, sequence_length, embedding_size)
        x *= tf.math.sqrt(tf.cast(self.embedding_size, tf.float32))
        x += self.positional_encoding[:, :sequence_length, :]
        x = self.dropout(x, training=training)
        for layer_index in range(self.layers_count):
            x = self.encoder_layers[layer_index](x, training, mask)
        return x  # (batch_size, input_seq_len, d_model)


class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(
            self, layers_count, embedding_size, heads_count,
            feed_forward_hidden_size, input_vocabulary_size,
            maximum_position_encoding, dropout_rate=0.1):
        super().__init__()
        self.embedding_size = embedding_size
        self.layers_count = layers_count
        self.embedding = tf.keras.layers.Embedding(
            input_vocabulary_size, embedding_size)
        self.positional_encoding = positional_encoding(
            maximum_position_encoding, embedding_size)
        self.decoder_layers = [
            DecoderLayer(
                embedding_size, heads_count, feed_forward_hidden_size,
                dropout_rate)
            for _ in range(layers_count)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(
            self, x, encoder_output, training, look_ahead_mask, padding_mask):
        sequence_length = tf.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.embedding_size, tf.float32))
        x += self.positional_encoding[:, :sequence_length, :]
        x = self.dropout(x, training=training)
        for layer_index in range(self.layers_count):
            x, block1, block2 = self.decoder_layers[layer_index](
                x, encoder_output, training, look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer{layer_index+1}_block1'] = block1
            attention_weights[f'decoder_layer{layer_index+1}_block2'] = block2
        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(keras.Model):
    def __init__(
            self, layers_count, embedding_size, heads_count,
            feed_forward_hidden_size, input_vocabulary_size,
            target_vocabulary_size, maximum_input_length,
            maximum_target_length, droput_rate=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(
            layers_count, embedding_size, heads_count,
            feed_forward_hidden_size, input_vocabulary_size,
            maximum_input_length, droput_rate)
        self.decoder = TransformerDecoder(
            layers_count, embedding_size, heads_count,
            feed_forward_hidden_size, target_vocabulary_size,
            maximum_target_length, droput_rate)
        self.final_layer = tf.keras.layers.Dense(target_vocabulary_size)

    def call(self, inputs, training):
        # Keras models prefer if you pass all your inputs in the first argument
        input, target = inputs
        encoder_padding_mask, look_ahead_mask, decoder_padding_mask = (
            self.create_masks(input, target))
        # (batch_size, input_sequence_length, embedding_size)
        encoder_output = self.encoder(input, training, encoder_padding_mask)
        # (batch_size, target_sequence_length, embedding_size)
        decoder_output, attention_weights = self.decoder(
            target, encoder_output, training, look_ahead_mask,
            decoder_padding_mask)
        # (batch_size, target_sequence_length, target_vocabulary_size)
        final_output = self.final_layer(decoder_output)
        return final_output, attention_weights

    def create_masks(self, input, target):
        # Encoder padding mask
        encoder_padding_mask = create_padding_mask(input)
        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        decoder_padding_mask = create_padding_mask(input)
        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])
        decoder_target_padding_mask = create_padding_mask(target)
        look_ahead_mask = tf.maximum(
            decoder_target_padding_mask, look_ahead_mask)
        return encoder_padding_mask, look_ahead_mask, decoder_padding_mask


def build_model(transformer):
    """
        Put some placeholder data through the transformer to initialise its
        shape to make it ready to load weights etc.
    """
    transformer([tf.zeros([64,1]),tf.zeros([64,1])])
