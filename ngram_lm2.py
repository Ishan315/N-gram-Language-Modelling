# preprocessing tools can be imported however, write code to compute probs
import numpy as np
import pandas as pd
from collections import Counter


class NgramLanguageModel:
    def __init__(self):
        pass
    def preprocess_text(self, path):
        with open(path, 'r') as file:
            lines = file.readlines()
        # Create a DataFrame from the list of lines
        df = pd.DataFrame(lines, columns=['review'])
        # Optionally, strip any extra whitespace or newline characters
        df['review'] = df['review'].str.strip()
        df["review"] = df["review"].apply(lambda x: x.lower())
        df["review"] = df["review"].apply(lambda x: "<s> " + x + " </s>")
        df["tokens"] = df["review"].apply(lambda x: x.split(" "))

        return df

    def preprocess_rare_words_unigram(self, unigram_counter, threshold=1):
        unigram_counter["<UNK>"] = 0

        for key, value in unigram_counter.items():
            if value <= threshold and key != "<UNK>":
                unigram_counter[key] = 0
                unigram_counter["<UNK>"] += 1

        processed_unigram_counter = Counter({k: v for k, v in unigram_counter.items() if v != 0})

        return processed_unigram_counter

    def preprocess_rare_words_bigram(self, bigram_tokens, unigram_counter, threshold=1):
        default_word = "<UNK>"
        # bigram_vocab = [(default_word if unigram_counter[word1] <= threshold else word1,
        #                  default_word if unigram_counter[word2] <= threshold else word2)
        #                 for word1, word2 in bigram_tokens]

        bigram_counter = Counter(bigram_tokens)
        bigram_counter["<UNK>"] = 0

        for key, value in bigram_counter.items():
            if value <= threshold and key != "<UNK>":
                bigram_counter[key] = 0
                bigram_counter["<UNK>"] += 1

        processed_bigram_counter = Counter({k: v for k, v in unigram_counter.items() if v != 0})

        return processed_bigram_counter

    def create_bigram_pairs(self, record):
        og_tokens = record["tokens"]
        offset_tokens = (record["tokens"][1:]
            # + ["<bigram_end>"]
            )
        combined_tuples = list(zip(og_tokens, offset_tokens))
        return combined_tuples

    def create_counters(self, df):
        unigram_tokens = [item for sublist in df['tokens'] for item in sublist]
        unigram_counter = Counter(unigram_tokens)
        processed_unigram_counter = self.preprocess_rare_words_unigram(unigram_counter)

        df["bigrams"] = df.apply(lambda x: self.create_bigram_pairs(x), axis=1)
        bigram_tokens = [item for sublist in df['bigrams'] for item in sublist]
        processed_bigram_counter = self.preprocess_rare_words_bigram(bigram_tokens, unigram_counter)

        return processed_unigram_counter, processed_bigram_counter

    def train_model(self, train_path):
        train_df = self.preprocess_text(train_path)
        train_unigram_counter, train_bigram_counter = self.create_counters(train_df)
        return train_unigram_counter, train_bigram_counter

    def infer_language_model(self, test_path, train_unigram_counter, train_bigram_counter, k=1):
        test_df = self.preprocess_text(test_path)
        test_unigram_counter, test_bigram_counter = self.create_counters(test_df)
        train_vocab_size = len(train_unigram_counter.keys())
        # print(train_unigram_counter)
        # print(train_vocab_size)
        test_vocab_size = len(test_unigram_counter.keys())
        # print(test_vocab_size)

        # Unigram
        test_unigram_prob_dict = {
            key: ((train_unigram_counter[key] if key in train_unigram_counter.keys()
                   else train_unigram_counter["<UNK>"]) + k) / (
                        sum(train_unigram_counter.values()) + k * train_vocab_size) for key in test_unigram_counter.keys()}

        test_unigram_perplexity = self.evaluate_perplexity(test_unigram_prob_dict, test_vocab_size)

        # Bigram
        test_bigram_prob_dict = {item: ((train_bigram_counter[item] if item in train_bigram_counter.keys()
                   else train_bigram_counter["<UNK>"]) + k) /
                                  ((train_unigram_counter[item[0]] if item[0] in train_unigram_counter.keys()
                                    else train_unigram_counter["<UNK>"])
                                   + k * train_vocab_size) for item in test_bigram_counter.keys()}

        test_bigram_perplexity = self.evaluate_perplexity(test_bigram_prob_dict, test_vocab_size)

        return test_unigram_perplexity, test_bigram_perplexity

    def evaluate_perplexity(self, probability_dict, vocab_size):
        log_sum = 0
        for prob in probability_dict.values():
            log_sum += np.log2(prob)

        return 2 ** (-log_sum / vocab_size)

    def run(self, train_path="./data/train.txt",
            test_path="./data/val.txt"):
        # train
        train_unigram_counter, train_bigram_counter = self.train_model(train_path)

        # evaluate
        unigram_perplexity, bigram_perplexity = (
            self.infer_language_model(test_path, train_unigram_counter, train_bigram_counter))
        print("Unigram Perplexity: ")
        print(unigram_perplexity)
        print(" ")

        print("Bigram Perplexity: ")
        print(bigram_perplexity)


if __name__ == '__main__':
    n_gram_lm = NgramLanguageModel()
    n_gram_lm.run()
