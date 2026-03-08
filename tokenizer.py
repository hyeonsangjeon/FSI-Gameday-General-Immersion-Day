"""KoBertTokenizer — SentencePiece-based tokenizer for KoBERT."""
import logging
import os
import unicodedata
from shutil import copyfile

from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "tokenizer_78b3253a26.model",
    "vocab_txt": "vocab.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "HyeonSang/kobert-sentiment": "https://huggingface.co/HyeonSang/kobert-sentiment/resolve/main/tokenizer_78b3253a26.model",
    },
    "vocab_txt": {
        "HyeonSang/kobert-sentiment": "https://huggingface.co/HyeonSang/kobert-sentiment/resolve/main/vocab.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "HyeonSang/kobert-sentiment": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "HyeonSang/kobert-sentiment": {"do_lower_case": False},
}

SPIECE_UNDERLINE = "\u2581"


class KoBertTokenizer(PreTrainedTokenizer):
    """SentencePiece based tokenizer for KoBERT.

    Requires `sentencepiece <https://github.com/google/sentencepiece>`_.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        vocab_txt,
        do_lower_case=False,
        remove_space=True,
        keep_accents=False,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs,
    ):
        # Build vocab FIRST before calling super().__init__
        self.token2idx: dict[str, int] = {}
        self.idx2token: list[str] = []
        with open(vocab_txt, "r", encoding="utf-8") as f:
            for idx, token in enumerate(f):
                token = token.strip()
                self.token2idx[token] = idx
                self.idx2token.append(token)

        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

        try:
            import sentencepiece as spm
        except ImportError:
            logger.warning(
                "You need to install SentencePiece to use KoBertTokenizer: "
                "https://github.com/google/sentencepiece  "
                "pip install sentencepiece"
            )
            raise

        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file
        self.vocab_txt = vocab_txt

        self.sp_model = spm.SentencePieceProcessor(model_file=vocab_file)

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.idx2token)

    def get_vocab(self) -> dict[str, int]:
        return self.token2idx.copy()

    # ------------------------------------------------------------------
    # Pickle support
    # ------------------------------------------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        try:
            import sentencepiece as spm
        except ImportError:
            logger.warning(
                "You need to install SentencePiece to use KoBertTokenizer: "
                "https://github.com/google/sentencepiece  "
                "pip install sentencepiece"
            )
            raise
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.vocab_file)

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def preprocess_text(self, inputs: str) -> str:
        if self.remove_space:
            outputs = " ".join(inputs.strip().split())
        else:
            outputs = inputs
        outputs = outputs.replace("``", '"').replace("''", '"')

        if not self.keep_accents:
            outputs = unicodedata.normalize("NFKD", outputs)
            outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
        if self.do_lower_case:
            outputs = outputs.lower()

        return outputs

    def _tokenize(self, text, return_unicode=True, sample=False):
        """Tokenize a string."""
        text = self.preprocess_text(text)

        if not sample:
            pieces = self.sp_model.EncodeAsPieces(text)
        else:
            pieces = self.sp_model.SampleEncodeAsPieces(text, 64, 0.1)

        new_pieces: list[str] = []
        for piece in pieces:
            if len(piece) > 1 and piece[-1] == "," and piece[-2].isdigit():
                cur_pieces = self.sp_model.EncodeAsPieces(
                    piece[:-1].replace(SPIECE_UNDERLINE, "")
                )
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)

        return new_pieces

    # ------------------------------------------------------------------
    # ID ↔ Token conversion
    # ------------------------------------------------------------------

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) to an id using the vocab."""
        return self.token2idx.get(token, self.token2idx[self.unk_token])

    def _convert_id_to_token(self, index: int, return_unicode=True) -> str:
        """Converts an index (integer) to a token (string) using the vocab."""
        return self.idx2token[index]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Converts a sequence of tokens (sub-words) into a single string."""
        return "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()

    # ------------------------------------------------------------------
    # Special tokens
    # ------------------------------------------------------------------

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """Build model inputs: ``[CLS] X [SEP]`` or ``[CLS] A [SEP] B [SEP]``."""
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0, token_ids_1=None, already_has_special_tokens=False
    ):
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(
                map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0)
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """Save the sentencepiece vocabulary and vocab.txt to a directory."""
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return

        out_vocab_model = os.path.join(save_directory, VOCAB_FILES_NAMES["vocab_file"])
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_model):
            copyfile(self.vocab_file, out_vocab_model)

        index = 0
        out_vocab_txt = os.path.join(save_directory, VOCAB_FILES_NAMES["vocab_txt"])
        with open(out_vocab_txt, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.token2idx.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(out_vocab_txt)
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1

        return out_vocab_model, out_vocab_txt
