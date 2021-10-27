import tensorflow as tf

from distributed_training_transformer import english_portugese_dataset, \
    transformer_model
from distributed_training_transformer.tester import Tester
from distributed_training_transformer.transformer_model import Transformer


sentence = "muitas pessoas vieram à estação."
SAVED_WEIGHTS_PATH = 'saved_weights/2/model_weights'


layers_count = 4
embedding_size = 128
feed_forward_hidden_size = 512
heads_count = 8
dropout_rate = 0.1
INPUT_VOCABULARY_SIZE = (
    english_portugese_dataset.tokenizers().pt.get_vocab_size().numpy())
TARGET_VOCABULARY_SIZE = (
    english_portugese_dataset.tokenizers().en.get_vocab_size().numpy())

model = Transformer(
        layers_count=layers_count,
        embedding_size=embedding_size,
        heads_count=heads_count,
        feed_forward_hidden_size=feed_forward_hidden_size,
        input_vocabulary_size=INPUT_VOCABULARY_SIZE,
        target_vocabulary_size=TARGET_VOCABULARY_SIZE,
        maximum_input_length=1000,
        maximum_target_length=1000,
        droput_rate=dropout_rate)
transformer_model.build_model(model)
model.load_weights(saved_weights_path)

translator = Tester(english_portugese_dataset.tokenizers(), model)

translated_text, translated_tokens, attention_weights = translator(
    tf.constant(sentence))
print(translated_text)
