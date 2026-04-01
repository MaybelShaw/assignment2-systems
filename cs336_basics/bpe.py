import time
import regex as re
from pathlib import Path
import multiprocessing.pool
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    return train_bpe_with_python(input_path, vocab_size, special_tokens)


def train_bpe_with_python(input_path: str, vocab_size: int, special_tokens: list[str]):
    # 初始化
    print("初始化 vocab 和 token2id...")
    vocab: dict[int, bytes] = {}
    token2id = {}

    for i in range(256):
        vocab[len(vocab)] = bytes([i])
        token2id[bytes([i])] = len(token2id)

    for st in special_tokens:
        vocab[len(vocab)] = st.encode("utf-8")
        token2id[st.encode("utf-8")] = len(token2id)

    # print(vocab)
    # print(token2id)

    # exit(0)

    print("查找文件分块边界...")
    num_processes = max(1, multiprocessing.pool.cpu_count() - 1)
    chunk_num = num_processes * 16

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, chunk_num, b"<|endoftext|>")

    args_list = [(input_path, start, end, token2id, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

    # s = time.time()

    # 多进程pre-tokenization
    print("开始多进程预分词...")
    words = Counter()
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(process_chunk, args) for args in args_list]

        for future in as_completed(futures):
            words += future.result()  # 获取子进程返回值
    #         print(f"Chunk done, found {len(words)} words")
    # print("用时:",time.time()-s)
    # import pprint
    # pprint.pprint(words)
    # exit(0)

    # 计算pairs
    print("计算初始 pairs...")
    pair = Counter()
    index = defaultdict(set)
    pairs = Counter()
    for w, c in words.items():
        for a, b in zip(w, w[1:]):
            pairs[(a, b)] += c
            index[(a, b)].add(w)

    # 迭代合并
    print("开始 BPE 训练...")
    i = 0
    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size:
        # 获取出现频率最高的pair
        best_pair = max(pairs.items(), key=lambda x: (x[1], (vocab[x[0][0]], vocab[x[0][1]])))[0]

        # 生成新token
        new_token = vocab[best_pair[0]] + vocab[best_pair[1]]
        vocab[len(vocab)] = new_token
        new_id = len(token2id)
        token2id[new_token] = new_id
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        # 增量更新 pairs 和 words
        affected_words = index[best_pair].copy()
        for w in affected_words:
            c = words[w]
            new_w = []

            i = 0
            while i < len(w):
                if i < len(w) - 1 and (w[i], w[i + 1]) == best_pair:
                    new_w.append(new_id)
                    i += 2
                else:
                    new_w.append(w[i])
                    i += 1
            new_w = tuple(new_w)

            for a, b in zip(w, w[1:]):
                pairs[(a, b)] -= c
                if w in index[(a, b)]:
                    index[(a, b)].remove(w)

            for a, b in zip(new_w, new_w[1:]):
                pairs[(a, b)] += c
                index[(a, b)].add(new_w)

            words.pop(w)
            words[new_w] = c
        print("训练进度:", len(vocab), "/", vocab_size)
        #     break
        # break
        # i+=1
        # print("loop:",i)
        # print(vocab)
        # print(merges)
        # print(end="\n\n\n\n")
    # print("BPE 训练完成")
    # print("最终 vocab 大小:", len(vocab))
    # print("最终 merges 大小:", len(merges))
    # print(vocab)
    # print(merges)
    # if b"<|" in vocab.values():
    #     print("警告: vocab 中不包含 <| 之类的特殊符号，可能影响后续的 tokenization")

    return (vocab, merges)


def process_chunk(args):
    path, start, end, token2id, special_tokens = args  # 注意：不能传递文件对象，必须传路径
    with open(path, "rb") as f:
        f.seek(start)
        raw_chunk = f.read(end - start)
        # 换行符统一为\n，确保跨平台兼容性
        chunk = raw_chunk.replace(b"\r\n", b"\n").decode("utf-8", errors="ignore")

        pattern = "|".join(re.escape(st) for st in special_tokens)
        chunks = re.split(pattern, chunk)

        words = Counter()
        for chunk in chunks:
            for match in re.finditer(PAT, chunk):

                b = match.group().encode("utf-8")
                if b in token2id:
                    words[(token2id[b],)] += 1
                else:
                    words[tuple(token2id[bytes([ch])] for ch in b)] += 1
    print(f"Chunk done, found {len(words)} words")
    return words


def train_bpe_with_rust(input_path: str, vocab_size: int, special_tokens: list[str]):
    pass


def train_bpe_tinystories():
    vocab, merges = train_bpe(path("TinyStoriesV2-GPT4-train.txt"), 10000, ["<|endoftext|>"])


def path(file_name: str) -> str:
    return Path(__file__).resolve().parent.parent / "data" / file_name

def serialize(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], output_path: str = None):
    import json

    vocab_hex = {k: v.hex() for k, v in vocab.items()}
    merges_hex = [(a.hex(), b.hex()) for a, b in merges]

    with open(output_path or "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_hex, f, ensure_ascii=False, indent=4)

    with open(output_path or "merges.json", "w", encoding="utf-8") as f:
        json.dump(merges_hex, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    print("开始训练 BPE...")
    # vocab, merges = train_bpe(
    #     path("TinyStoriesV2-GPT4-valid.txt"),
    #     1000,
    #     ["<|endoftext|>"],
    # )
    vocab, merges = train_bpe(path("TinyStoriesV2-GPT4-train.txt"), 10000, ["<|endoftext|>"])
    print("训练完成")
    serialize(vocab, merges)
    print("已保存 bpe_vocab.json")

    # train_bpe_tinystories()
    # import pprint
    # pprint.pprint(vocab)
    # import os
    # print("逻辑核心数:", os.cpu_count())          # 含超线程
    # print("物理核心数:", os.cpu_count() // 2)  # 不含超线程
    # print("",os.name)
