import numpy as np
import pandas as pd
import re
from collections import Counter


class NgramLanguageModel:
    def __init__(self):
        self.enable_thresholding = True
        self.enable_smoothing = False
        self.rare_word_count_threshold = 1
        self.smoothing_k = 1

    def preprocess_text(self, path):
        """
        Preprocesses text data by reading a file, cleaning the text, and tokenizing each sentence.

        Args:
        - path (str): The path to the text file to be preprocessed.

        Returns:
        - pd.DataFrame: A DataFrame with the original text and tokenized sentences.
        """

        # Open the file at the given path and read its contents line by line
        with open(path, 'r') as file:
            lines = file.readlines()  # Read all lines in the file into a list

        # Create a DataFrame from the list of lines, with 'review' as the column name
        df = pd.DataFrame(lines, columns=['review'])

        # Strip any extra whitespace or newline characters from each review
        df['review'] = df['review'].str.strip()

        # Convert all text to lowercase to standardize the text
        df["review"] = df["review"].apply(lambda x: x.lower())

        # Add start-of-sentence (<s>) and end-of-sentence (</s>) tags to each review
        df["review"] = df["review"].apply(lambda x: "<s> " + x + " </s>")

        # Tokenize each review by splitting the string into a list of words (tokens)
        df["tokens"] = df["review"].apply(lambda x: x.split(" "))

        return df  # Return the DataFrame with processed text

    def preprocess_rare_words_unigram(self, unigram_counter):
        """
        Replaces rare words in a unigram counter with a special token <UNK> (unknown).

        Args:
        - unigram_counter (Counter): A Counter object where keys are unigrams (words) and values are their frequencies.

        Returns:
        - Counter: A modified unigram counter where rare words have been replaced with <UNK> and removed.
        """

        # Initialize the count for the <UNK> token in the unigram counter
        unigram_counter["<UNK>"] = 0

        # Iterate through the unigrams and their counts
        for key, value in unigram_counter.items():
            # If the count of the unigram is below the rare_word_count_threshold and it's not <UNK>
            if value <= self.rare_word_count_threshold and key != "<UNK>":
                # Set the unigram count to 0 (effectively removing it)
                unigram_counter[key] = 0
                # Increment the count of <UNK> by the frequency of the rare word
                unigram_counter["<UNK>"] += 1

        # Create a new Counter that only includes unigrams with non-zero counts
        processed_unigram_counter = Counter({k: v for k, v in unigram_counter.items() if v != 0})

        return processed_unigram_counter

    def preprocess_rare_words_bigram(self, bigram_counter, unigram_counter):
        bigram_counter["<UNK>"] = 0

        for key, value in bigram_counter.items():
            if value <= self.rare_word_count_threshold and key != "<UNK>":
                bigram_counter[key] = 0
                bigram_counter["<UNK>"] += 1

        processed_bigram_counter = Counter({k: v for k, v in bigram_counter.items() if v != 0})

        return processed_bigram_counter

    def create_bigram_pairs(self, record):
        og_tokens = record["tokens"]
        offset_tokens = record["tokens"][1:]
        combined_tuples = list(zip(og_tokens, offset_tokens))
        return combined_tuples

    def create_counters(self, df):
        unigram_tokens = [item for sublist in df['tokens'] for item in sublist]
        unigram_counter = Counter(unigram_tokens)

        df["bigrams"] = df.apply(lambda x: self.create_bigram_pairs(x), axis=1)
        bigram_tokens = [item for sublist in df['bigrams'] for item in sublist]
        bigram_counter = Counter(bigram_tokens)

        if self.enable_thresholding:
            processed_unigram_counter = self.preprocess_rare_words_unigram(unigram_counter)
            processed_bigram_counter = self.preprocess_rare_words_bigram(bigram_counter, unigram_counter)

        else:
            processed_unigram_counter = unigram_counter
            processed_bigram_counter = bigram_counter

        return processed_unigram_counter, processed_bigram_counter

    def train_model(self, train_path):
        train_df = self.preprocess_text(train_path)
        train_unigram_counter, train_bigram_counter = self.create_counters(train_df)
        train_vocab_size = len(train_unigram_counter.keys())
        print("Train Vocabulary Size: {}".format(train_vocab_size))
        print("Train Unigram Counter count: {}".format(sum(train_unigram_counter.values())))
        print("Train Bigram Counter Size: {}".format(len(train_bigram_counter.keys())))
        print("Train Bigram Counter count: {}".format(sum(train_bigram_counter.values())))

        train_unigram_probabilities = {key: value / sum(train_unigram_counter.values()) for key, value in train_unigram_counter.items()}
        train_bigram_probabilities = {key: (value / train_unigram_counter[key[0]]
                                            ) if train_unigram_counter[key[0]] != 0 else 1e-6
                              for key, value in train_bigram_counter.items()}

        train_unigram_perplexity = self.evaluate_perplexity(train_unigram_probabilities, train_vocab_size)
        train_bigram_perplexity = self.evaluate_perplexity(train_bigram_probabilities, train_vocab_size)

        return train_unigram_counter, train_bigram_counter, train_unigram_perplexity, train_bigram_perplexity

    def infer_language_model(self, test_path, train_unigram_counter, train_bigram_counter):
        test_df = self.preprocess_text(test_path)
        test_unigram_counter, test_bigram_counter = self.create_counters(test_df)
        train_vocab_size = len(train_unigram_counter.keys())
        # print(train_unigram_counter)
        # print(train_vocab_size)
        test_vocab_size = len(test_unigram_counter.keys())
        # print(test_vocab_size)

        if self.enable_smoothing:
            # Unigram
            test_unigram_prob_dict = {
                key: ((train_unigram_counter[key] if key in train_unigram_counter.keys()
                       else train_unigram_counter["<UNK>"]) + self.smoothing_k) / (
                             sum(train_unigram_counter.values()) +
                             self.smoothing_k * train_vocab_size) for key in test_unigram_counter.keys()}

        else:
            test_unigram_prob_dict = {
                key: (((train_unigram_counter[key] if key in train_unigram_counter.keys()
                       else train_unigram_counter["<UNK>"])) /
                         sum(train_unigram_counter.values())) if sum(train_unigram_counter.values()) != 0 else
                         1e-6 for key in test_unigram_counter.keys()}

        test_unigram_perplexity = self.evaluate_perplexity(test_unigram_prob_dict, test_vocab_size)

        if self.enable_smoothing:
            # Bigram
            test_bigram_prob_dict = {item: ((train_bigram_counter[item] if item in train_bigram_counter.keys()
                                             else train_bigram_counter["<UNK>"]) + self.smoothing_k) /
                                           ((train_unigram_counter[item[0]] if item[0] in train_unigram_counter.keys()
                                             else train_unigram_counter["<UNK>"])
                                            + self.smoothing_k * train_vocab_size) for item in
                                     test_bigram_counter.keys()}

        else:
            test_bigram_prob_dict = {item: (((train_bigram_counter[item] if item in train_bigram_counter.keys()
                                             else train_bigram_counter["<UNK>"])) /
                                           train_unigram_counter.get(item[0], train_unigram_counter[
                                               "<UNK>"])) if train_unigram_counter.get(item[0], train_unigram_counter[
                                               "<UNK>"]) != 0 else 1e-6
                                     for item in
                                     test_bigram_counter.keys()}

        test_bigram_perplexity = self.evaluate_perplexity(test_bigram_prob_dict, test_vocab_size)

        return test_unigram_perplexity, test_bigram_perplexity

    def evaluate_perplexity(self, probability_dict, vocab_size):
        """
        Evaluates the perplexity of a language model based on the given probabilities of words.

        Args:
        - probability_dict (dict): A dictionary where keys are words (or tokens) and values are their corresponding probabilities.
        - vocab_size (int): The total number of unique words (or tokens) in the vocabulary.

        Returns:
        - float: The perplexity score, which indicates how well the probability distribution predicts the sample.
                 A lower perplexity indicates a better model.
        """
        log_sum = 0  # Initialize a variable to hold the sum of logarithms of probabilities

        # Loop over all probabilities in the dictionary and compute the sum of their log base 2
        for prob in probability_dict.values():
            log_sum += np.log2(prob)

        # Calculate perplexity using the formula: 2 ** (-log_sum / vocab_size)
        # This formula is derived from entropy: perplexity = 2^(-entropy)
        return 2 ** (-log_sum / vocab_size)

    def run_inference(self, train_path="./data/train.txt",
            test_path="./data/val.txt"):

        # train
        train_unigram_counter, train_bigram_counter, train_unigram_perplexity, train_bigram_perplexity = (
            self.train_model(train_path))
        print("Train Set:")
        print("Unigram Perplexity: ")
        print(train_unigram_perplexity)
        print(" ")

        print("Bigram Perplexity: ")
        print(train_bigram_perplexity)
        print(" ")

        # evaluate
        unigram_perplexity, bigram_perplexity = (
            self.infer_language_model(test_path, train_unigram_counter, train_bigram_counter))
        print("Validation Set:")
        print("Unigram Perplexity: ")
        print(unigram_perplexity)
        print(" ")

        print("Bigram Perplexity: ")
        print(bigram_perplexity)


if __name__ == '__main__':
    n_gram_lm = NgramLanguageModel()
    n_gram_lm.run_inference()
