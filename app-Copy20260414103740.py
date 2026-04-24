import streamlit as st
import tensorflow as tf
from tensorflow import keras
import pickle
import re
import string
from keras import layers, ops

# SETTINGS
MAX_SEQUENCE_LENGTH = 60

# CUSTOM STANDARDIZATION
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    punctuation_to_remove = string.punctuation.replace("[", "").replace("]", "")
    return tf.strings.regex_replace(
        lowercase, f"[{re.escape(punctuation_to_remove)}]", ""
    )

# CUSTOM LAYERS
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.supports_masking = True

        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            mask_zero=True
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length,
            output_dim=embed_dim
        )

    def call(self, inputs):
        length = ops.shape(inputs)[1]
        positions = ops.arange(0, length, 1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        embedded_positions = ops.expand_dims(embedded_positions, axis=0)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return ops.not_equal(inputs, 0)


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential([
            layers.Dense(dense_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        attention_output = self.attention(
            query=inputs,
            value=inputs,
            key=inputs
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential([
            layers.Dense(latent_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()

    def call(self, inputs, encoder_outputs, mask=None):
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]

        i = ops.arange(seq_len)[:, None]
        j = ops.arange(seq_len)
        causal_mask = ops.cast(i >= j, "int32")
        causal_mask = ops.reshape(causal_mask, (1, seq_len, seq_len))
        causal_mask = ops.tile(causal_mask, (batch_size, 1, 1))

        combined_mask = causal_mask[:, None, :, :]
        if mask is not None:
            padding_mask = ops.cast(mask[:, None, None, :], "int32")
            combined_mask = ops.minimum(combined_mask, padding_mask)

        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=combined_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

# LOAD VOCAB + MODEL
@st.cache_resource
def load_artifacts():
    with open("en_vocab.pkl", "rb") as f:
        en_vocab = pickle.load(f)

    with open("fr_vocab.pkl", "rb") as f:
        fr_vocab = pickle.load(f)

    en_vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=len(en_vocab),
        output_mode="int",
        output_sequence_length=MAX_SEQUENCE_LENGTH
    )
    en_vectorizer.set_vocabulary(en_vocab)

    fr_vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=len(fr_vocab),
        output_mode="int",
        output_sequence_length=MAX_SEQUENCE_LENGTH,
        standardize=custom_standardization
    )
    fr_vectorizer.set_vocabulary(fr_vocab)

    fr_index_lookup = dict(zip(range(len(fr_vocab)), fr_vocab))

    model = keras.models.load_model(
        "transformer_translation.keras",
        custom_objects={
            "PositionalEmbedding": PositionalEmbedding,
            "TransformerEncoder": TransformerEncoder,
            "TransformerDecoder": TransformerDecoder,
        }
    )

    return model, en_vectorizer, fr_vectorizer, fr_index_lookup

model, en_vectorizer, fr_vectorizer, fr_index_lookup = load_artifacts()

# DECODE FUNCTION
def decode_sequence(input_sentence):
    tokenized_input_sentence = en_vectorizer([input_sentence])
    decoded_sentence = "[start]"

    for i in range(MAX_SEQUENCE_LENGTH - 1):
        tokenized_target_sentence = fr_vectorizer([decoded_sentence])[:, :-1]

        predictions = model.predict(
            [tokenized_input_sentence, tokenized_target_sentence],
            verbose=0
        )

        sampled_token_index = int(tf.argmax(predictions[0, i, :]))
        sampled_token = fr_index_lookup.get(sampled_token_index, "")

        if sampled_token == "[end]":
            break

        if sampled_token not in ["", "[start]"]:
            decoded_sentence += " " + sampled_token

    return decoded_sentence.replace("[start]", "").strip()

# STREAMLIT UI
st.set_page_config(page_title="English to French Translator", page_icon="🌍")

st.title("English to French Translator")
st.write("Enter an English sentence and get a French translation.")

user_input = st.text_area("Enter English text:", placeholder="Type your sentence here...")

if st.button("Translate"):
    if user_input.strip():
        translation = decode_sequence(user_input)
        st.subheader("French Translation")
        st.success(translation)
    else:
        st.warning("Please enter some text first.")