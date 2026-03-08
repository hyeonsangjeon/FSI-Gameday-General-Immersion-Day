"""KoBERT Sentiment model loading and inference."""
import os
import logging

# Must be set BEFORE importing tensorflow so it uses tf-keras (Keras 2)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download
from transformers import BertConfig, TFBertModel
from tensorflow.keras import optimizers, regularizers

from tokenizer import KoBertTokenizer

logger = logging.getLogger(__name__)

# TF log level
tf.get_logger().setLevel(logging.ERROR)


def create_sentiment_bert(
    config: BertConfig,
    learning_rate: float = 1e-6,
    seq_len: int = 32,
    dropout: float = 0.01,
) -> tf.keras.Model:
    """Create sentiment classification model from BertConfig (no torch needed)."""
    model = TFBertModel(config)

    token_inputs = tf.keras.layers.Input((seq_len,), dtype=tf.int32, name="input_word_ids")
    mask_inputs = tf.keras.layers.Input((seq_len,), dtype=tf.int32, name="input_masks")
    segment_inputs = tf.keras.layers.Input((seq_len,), dtype=tf.int32, name="input_segment")

    bert_outputs = model(
        input_ids=token_inputs,
        attention_mask=mask_inputs,
        token_type_ids=segment_inputs,
    )
    bert_outputs = bert_outputs[1]
    sentiment_drop = tf.keras.layers.Dropout(dropout)(bert_outputs)
    sentiment_output = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        bias_regularizer=regularizers.l2(1e-4),
        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
        activity_regularizer=regularizers.l2(1e-5),
    )(sentiment_drop)

    sentiment_model = tf.keras.Model(
        [token_inputs, mask_inputs, segment_inputs], sentiment_output
    )
    opt = optimizers.Adam(learning_rate=learning_rate)
    sentiment_model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy())
    return sentiment_model


class SentimentPredictor:
    """Encapsulates model + tokenizer for inference."""

    def __init__(
        self,
        model_repo: str = "HyeonSang/kobert-sentiment",
        seq_len: int = 32,
    ):
        self.seq_len = seq_len
        self.model_repo = model_repo

        # GPU vs CPU strategy
        if os.path.exists("/app"):  # Docker
            self.strategy = tf.distribute.MirroredStrategy(
                devices=["/gpu:0"],
                cross_device_ops=tf.distribute.HierarchicalCopyAllReduce(),
            )
        else:
            self.strategy = tf.distribute.get_strategy()

        self._load_tokenizer()
        self._load_model()
        logger.info("SentimentPredictor initialized successfully")

    def _load_tokenizer(self):
        vocab_file = hf_hub_download(
            repo_id=self.model_repo, filename="tokenizer_78b3253a26.model"
        )
        vocab_txt = hf_hub_download(
            repo_id=self.model_repo, filename="vocab.txt"
        )
        self.tokenizer = KoBertTokenizer(vocab_file=vocab_file, vocab_txt=vocab_txt)

    def _load_model(self):
        config = BertConfig.from_pretrained(self.model_repo)

        # Try fine-tuned full model first, fall back to base BERT weights
        try:
            model_file = hf_hub_download(
                repo_id=self.model_repo, filename="fsi_comment_sentiment_model.h5"
            )
        except Exception:
            local_path = os.path.join(
                os.path.dirname(__file__), "data", "fsi_comment_sentiment_model.h5"
            )
            if os.path.exists(local_path):
                model_file = local_path
                logger.info("Using local fine-tuned model: %s", local_path)
            else:
                model_file = hf_hub_download(
                    repo_id=self.model_repo, filename="tf_model.h5"
                )
                logger.warning(
                    "Fine-tuned model not found — using base BERT weights only. "
                    "Predictions will be inaccurate."
                )

        with self.strategy.scope():
            self.model = create_sentiment_bert(config=config, seq_len=self.seq_len)
            self.model.load_weights(model_file, by_name=True, skip_mismatch=True)

    def _tokenize(self, text: str):
        token = self.tokenizer.encode(
            text, max_length=self.seq_len, padding="max_length", truncation=True
        )
        num_zeros = token.count(0)
        mask = [1] * (self.seq_len - num_zeros) + [0] * num_zeros
        segment = [0] * self.seq_len
        return [np.array([token]), np.array([mask]), np.array([segment])]

    def predict(self, text: str) -> dict:
        inputs = self._tokenize(text)
        with self.strategy.scope():
            raw = self.model.predict(inputs, verbose=0)
        score = float(np.ravel(raw)[0])
        is_positive = round(score) == 1
        return {
            "text": text,
            "sentiment": "긍정" if is_positive else "부정",
            "confidence": round((score if is_positive else 1 - score) * 100, 2),
            "raw_score": round(score, 4),
        }
