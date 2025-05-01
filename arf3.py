import numpy as np
import pandas as pd
import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox
from collections import defaultdict
import math
from tqdm import tqdm  # プログレスバー用

STOPWORDS = set([
    "the", "is", "in", "at", "on", "and", "a", "an", "of", "to", "for", "with",
    "that", "by", "this", "as", "are", "was"
])

USE_STOPWORDS = False

def calculate_arf(corpus_length, positions, num_offsets=10):
    freq = len(positions)
    if freq == 0:
        return 0.0
    block_size = corpus_length / freq
    offset_step = block_size / num_offsets
    arf_values = []
    for i in range(num_offsets):
        offset = i * offset_step
        blocks = set()
        for pos in positions:
            block_index = int((pos + offset) / block_size)
            blocks.add(block_index)
        arf_values.append(len(blocks))
    return np.mean(arf_values)

def compute_stats(corpus_tokens, documents, filenames, num_offsets=10):
    corpus_length = len(corpus_tokens)
    word_positions = defaultdict(list)
    word_document_occurrence = defaultdict(set)

    for idx, word in enumerate(corpus_tokens):
        if not USE_STOPWORDS or word not in STOPWORDS:
            word_positions[word].append(idx)

    for doc_index, doc_tokens in enumerate(documents):
        unique_words = set(doc_tokens)
        for word in unique_words:
            if not USE_STOPWORDS or word not in STOPWORDS:
                word_document_occurrence[word].add(doc_index)

    total_docs = len(documents)
    results = []

    print("統計情報を計算中...")
    for word, positions in tqdm(word_positions.items(), desc="Processing words"):
        freq = len(positions)
        arf = calculate_arf(corpus_length, positions, num_offsets)
        tf_total = freq / corpus_length
        df = len(word_document_occurrence[word])
        idf = math.log((total_docs + 1) / (df + 1)) + 1

        tf_doc = []
        tfidf_doc = []
        for doc_tokens in documents:
            tf = doc_tokens.count(word) / len(doc_tokens)
            tf_doc.append(tf)
            tfidf_doc.append(tf * idf)

        row = [word, arf, freq, df, idf] + tf_doc + tfidf_doc
        results.append(row)

    results.sort(key=lambda x: -x[2])  # 頻度順
    return results

def select_files(title):
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title=title,
        filetypes=[("Text Files", "*.txt")],
        initialdir=os.getcwd()
    )
    return file_paths

def load_multiple_texts(file_paths):
    documents = []
    filenames = []

    if not file_paths:
        print("ファイルが選択されませんでした。処理を中断します。")
        return None, None, None

    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            tokens = re.findall(r"\b\w+\b", text.lower())
            documents.append(tokens)
            filenames.append(os.path.basename(file_path))

    corpus_tokens = [word for doc in documents for word in doc]
    return corpus_tokens, documents, filenames

def load_lemma_dict(lemma_file_path):
    if not lemma_file_path:
        print("Lemmaリストが選択されませんでした。処理を中断します。")
        return None

    lemma_dict = {}
    with open(lemma_file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(" -> ")
            if len(parts) == 2:
                lemma, variants = parts
                lemma_dict[lemma] = [v.strip().lower() for v in variants.split(",")]

    return lemma_dict

def convert_to_lemma(tokens, lemma_dict):
    reverse_dict = {}
    for lemma, variants in lemma_dict.items():
        for variant in variants:
            reverse_dict[variant] = lemma
    return [reverse_dict.get(token, token) for token in tokens]

def convert_documents_to_lemmas(documents, lemma_dict):
    reverse_dict = {}
    for lemma, variants in lemma_dict.items():
        for variant in variants:
            reverse_dict[variant] = lemma
    new_docs = []
    for doc in documents:
        new_docs.append([reverse_dict.get(token, token) for token in doc])
    return new_docs

def save_stats_to_csv(stats, filenames):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(
        title="CSVファイルの保存先と名前を指定してください",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        initialfile="tfidf_arf_results.csv"
    )
    if not file_path:
        print("保存がキャンセルされました。")
        return

    # 列名
    columns = ["word", "arf", "frequency", "doc_frequency", "idf"]
    columns += [f"tf_{name}" for name in filenames]
    columns += [f"tf*idf_{name}" for name in filenames]

    df = pd.DataFrame(stats, columns=columns)
    df.to_csv(file_path, index=False)
    print(f"CSVを出力しました！保存先: {file_path}")

# ===== 実行フロー =====

lemma_file_path = filedialog.askopenfilename(
    title="LemmaリストTXTファイルを選択してください",
    filetypes=[("Text Files", "*.txt")]
)
lemma_dict = load_lemma_dict(lemma_file_path)

file_paths = select_files("分析対象のTXTファイルを選択してください")
corpus_tokens, documents, filenames = load_multiple_texts(file_paths)

USE_STOPWORDS = messagebox.askyesno("ストップワード除去", "ストップワードを除去しますか？")

if corpus_tokens and lemma_dict:
    lemma_tokens = convert_to_lemma(corpus_tokens, lemma_dict)
    documents = convert_documents_to_lemmas(documents, lemma_dict)
    stats = compute_stats(lemma_tokens, documents, filenames)
    save_stats_to_csv(stats, filenames)
    print("処理が完了しました！")
