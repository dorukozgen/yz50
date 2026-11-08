
import os
import multiprocessing as mp
import numpy as np
import tiktoken

from datasets import load_dataset
from tqdm import tqdm

# ------------------------------------------
# local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
local_dir = "dataset"

# Test için 1M tokenlık shard
# Production için int(1e8) kullanabilirsin.
shard_size = int(1e6)

max_docs = 50_000

enc = None
eot = None


def init_tokenizer():
    global enc, eot
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token


def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))

    tokens_np = np.array(tokens)

    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all()

    return tokens_np.astype(np.uint16)


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


def main():
    # Create local directory
    data_cache_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        local_dir
    )
    os.makedirs(data_cache_dir, exist_ok=True)

    # Load dataset
    fw = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=remote_name,
        split="train",
        streaming=True
    )
    # fw = load_dataset(
    #         "tascib/turkish-llm-dataset",
    #         split="train",
    #         streaming=True
    #     )
    

    fw = fw.take(max_docs)

    # Multiprocessing
    nprocs = max(1, (os.cpu_count() or 2) // 2)

    print(f"Using {nprocs} processes")

    with mp.Pool(
        processes=nprocs,
        initializer=init_tokenizer
    ) as pool:

        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0

        progress_bar = tqdm(
            unit="tokens",
            desc="Tokenizing"
        )

        for tokens in pool.imap(tokenize, fw, chunksize=16):

            offset = 0

            # Handle documents that cross shard boundaries
            while offset < len(tokens):

                remaining = shard_size - token_count
                n = min(remaining, len(tokens) - offset)

                all_tokens_np[token_count:token_count+n] = tokens[offset:offset+n]

                token_count += n
                offset += n

                progress_bar.update(n)

                # Shard is full
                if token_count == shard_size:

                    split = "val" if shard_index == 0 else "train"

                    filename = os.path.join(
                        data_cache_dir,
                        f"dataset_{split}_{shard_index:06d}"
                    )

                    write_datafile(filename, all_tokens_np)

                    shard_index += 1
                    token_count = 0

        # Write remaining tokens
        if token_count > 0:

            split = "val" if shard_index == 0 else "train"

            filename = os.path.join(
                data_cache_dir,
                f"dataset_{split}_{shard_index:06d}"
            )

            write_datafile(
                filename,
                all_tokens_np[:token_count]
            )

        progress_bar.close()

    print("Dataset tokenization completed!")


if __name__ == "__main__":
    mp.freeze_support()
    main()