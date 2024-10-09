import numpy as np
import pandas as pd
from collections import Counter


# 1. Preprocessing functions
def preprocess_text(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Create DataFrame and process reviews
    df = pd.DataFrame(lines, columns=['review'])
    df['review'] = df['review'].str.strip().str.lower()
    df['review'] = df['review'].apply(lambda x: "<s> " + x + " </s>")
    df['tokens'] = df['review'].apply(lambda x: x.split(" "))
    return df


def get_vocab(df):
    return [item for sublist in df['tokens'] for item in sublist]


# 2. Vocabulary creation functions
def create_unigram_counter(df, threshold=2):
    vocab = get_vocab(df)
    unigram_counter = Counter(vocab)
    unigram_counter["<UNK>"] = 0

    for key, value in unigram_counter.items():
        if value <= threshold and key != "<UNK>":
            unigram_counter[key] = 0
            unigram_counter["<UNK>"] += 1

    unigram_counter = Counter({k: v for k, v in unigram_counter.items() if v != 0})
    return unigram_counter


# 3. Bigram handling functions
def create_bigram_pairs(record):
    og_tokens = record["tokens"]
    offset_tokens = record["tokens"][1:]
    combined_tuples = list(zip(og_tokens, offset_tokens))
    return combined_tuples


def create_bigram_vocab(df, unigram_counter, threshold=2):
    bigram_vocab = [item for sublist in df['bigrams'] for item in sublist]
    bigram_vocab = [(word1 if unigram_counter[word1] > threshold else "<UNK>",
                     word2 if unigram_counter[word2] > threshold else "<UNK>")
                    for word1, word2 in bigram_vocab]
    return bigram_vocab


# 4. Probability calculation functions
def calculate_unigram_probs(unigram_counter, vocab_size, k=0.09):
    total_count = sum(unigram_counter.values())
    unigram_probs = {key: (value + k) / (total_count + k * vocab_size)
                     for key, value in unigram_counter.items()}
    return unigram_probs


def calculate_bigram_probs(bigram_counter, unigram_counter, vocab_size, k=0.09):
    bigram_probs = {key: (value + k) / (unigram_counter[key[0]] + k * vocab_size)
                    for key, value in bigram_counter.items()}
    return bigram_probs


# 5. Perplexity evaluation function
def evaluate_perplexity(probs, N):
    log_sum = np.sum([np.log2(prob) for prob in probs.values()])
    return 2 ** (-log_sum / N)


# 6. Validation function
def validate_model(val_df, bigram_counter, unigram_counter, vocab_size, k=0.09):
    val_vocab = [item for sublist in val_df['tokens'] for item in sublist]
    val_bigram_vocab = [item for sublist in val_df['bigrams'] for item in sublist]

    val_bigram_probs = {item: ((bigram_counter[item] + k) /
                               (unigram_counter[item[0]] if item[0] in unigram_counter else unigram_counter["<UNK>"])
                               + k * vocab_size) for item in val_bigram_vocab}

    val_unigram_counter = Counter(val_vocab)
    val_unigram_probs = {key: ((unigram_counter.get(key, unigram_counter["<UNK>"]) + k)
                               / (sum(unigram_counter.values()) + k * vocab_size))
                         for key in val_unigram_counter.keys()}

    unigram_perplexity = evaluate_perplexity(val_unigram_probs, len(set(val_vocab)))
    bigram_perplexity = evaluate_perplexity(val_bigram_probs, len(set(val_vocab)))
    return unigram_perplexity, bigram_perplexity


# Main function to process the training data
def main():
    # 1. Preprocessing
    df = preprocess_text('A1_DATASET/train.txt')
    df["bigrams"] = df.apply(lambda x: create_bigram_pairs(x), axis=1)

    # 2. Vocabulary creation
    unigram_counter, vocab = create_unigram_counter(df)
    bigram_vocab = create_bigram_vocab(df, unigram_counter)

    # 3. Bigram counting and smoothing
    bigram_counter = Counter(bigram_vocab)
    vocab_size = len(unigram_counter)
    bigram_probs = calculate_bigram_probs(bigram_counter, unigram_counter, vocab_size)

    # Validation on validation set
    val_df = preprocess_text('A1_DATASET/val.txt')
    val_df["bigrams"] = val_df.apply(lambda x: create_bigram_pairs(x), axis=1)
    unigram_perplexity, bigram_perplexity = validate_model(val_df, bigram_counter, unigram_counter, vocab_size)

    print(f'Unigram Perplexity: {unigram_perplexity}')
    print(f'Bigram Perplexity: {bigram_perplexity}')


if __name__ == "__main__":
    main()
