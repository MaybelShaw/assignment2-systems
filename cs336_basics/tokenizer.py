import json
import regex as re
from pathlib import Path
from collections.abc import Iterable


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.token2id = {v: k for k, v in vocab.items()}
        self.merges = merges

        # 最长匹配，优先匹配长的special token,防止短的special token是长的special token的子串
        self.special_tokens = sorted(special_tokens,key=len,reverse=True) if special_tokens is not None else []

        # 使用map加速查找，o(n)->o(1)
        self.merges_rank = {pair: i for i, pair in enumerate(merges)}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        with open(vocab_filepath, "r", encoding="utf-8") as vf:
            vocab_hex = json.load(vf)
        vocab = {int(k): bytes.fromhex(v) for k, v in vocab_hex.items()}

        with open(merges_filepath, "r", encoding="utf-8") as mf:
            merges_hex = json.load(mf)
        merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in merges_hex]

        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            pattern = "|".join(re.escape(st) for st in self.special_tokens)
            chunks = re.split(f"({pattern})", text)
            for i in range(len(chunks)):
                if chunks[i] in self.special_tokens:
                    chunks[i] = self.token2id[chunks[i].encode("utf-8")]
            chunks = [chunk for chunk in chunks if chunk != ""]
        else:
            chunks = [text]


        PAT = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        tokens = []
        for chunk in chunks:
            if isinstance(chunk, int):
                tokens.append(chunk)
                continue
            for match in re.finditer(PAT, chunk):
                word = match.group()
                pre_token = [self.token2id[bytes([b])] for b in word.encode("utf-8")]

                while True:
                    i = 0
                    pairs = []
                    while i < len(pre_token) - 1:
                        pair = (self.vocab[pre_token[i]], self.vocab[pre_token[i + 1]])
                        if pair in self.merges_rank:
                            pairs.append((self.merges_rank[pair], pair, i))
                        i += 1
                    if not pairs:
                        break

                    # 选择排名最高的pair进行合并
                    _, best_pair, idx = min(pairs, key=lambda x: x[0])
                    new_token = self.token2id[best_pair[0] + best_pair[1]]
                    pre_token = pre_token[:idx] + [new_token] + pre_token[idx + 2 :]

                tokens.extend(pre_token)

        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        bytes_list = [self.vocab[i] for i in ids]
        return b"".join(bytes_list).decode("utf-8", errors="replace")


def path(file_name: str) -> str:
    return Path(__file__).resolve().parent.parent / "data" / file_name



if __name__ == "__main__":
    tokenizer = Tokenizer.from_files(
        path("vocab.json"), path("merges.json"), ["<|endoftext|>"]
    )
    text = "Hello, how are you?"
    ids = tokenizer.encode(text)
    print("Encoded IDs:", ids)
    decoded_text = tokenizer.decode(ids)
    print("Decoded text:", decoded_text)