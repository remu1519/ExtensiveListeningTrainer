import tkinter as tk
from tkinter import messagebox
import wikipediaapi
import re
from whisperspeech.pipeline import Pipeline
from dotenv import load_dotenv
import os
from pydub import AudioSegment
import torch
import threading

# .envファイルから環境変数をロード
load_dotenv()

CONTACT_EMAIL = os.getenv("CONTACT_EMAIL")

# CUDA（GPU）の利用確認
def check_cuda():
    if not torch.cuda.is_available():
        raise BaseException("This application requires CUDA. Please run this on a machine with a GPU.")
    print("CUDA is available. Proceeding with the pipeline.")

# WhisperSpeechのパイプライン初期化
def init_pipeline():
    return Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-small-en+pl.model')

# ファイル名に使用できない文字を除去する関数
def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "", filename)

# 出力ディレクトリを作成する関数
def ensure_output_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# テキストを一文ごとに分割する関数
def split_text_into_sentences(text):
    sentences = re.split(r'(?:\n|(?<=[.!?]))+', text.strip())  # 改行または文末記号で分割
    return [sentence for sentence in sentences if sentence]

# ファイルパスを生成する関数
def create_audio_filepath(base_dir, prefix, section_title):
    sanitized_title = sanitize_filename(section_title)
    return os.path.join(base_dir, f"{prefix}_{sanitized_title}.wav")

# 音声ファイル生成の処理
def generate_audio(pipe, sentences, output_file):
    combined_audio = AudioSegment.empty()
    silence = AudioSegment.silent(duration=400)  # 0.4秒の無音

    for idx, sentence in enumerate(sentences):
        if not sentence.strip():
            continue

        temp_file = os.path.join("out", "temp", f"temp_{idx}.wav")
        ensure_output_directory(os.path.dirname(temp_file))

        try:
            # 各文を音声化して一時ファイルに保存
            pipe.generate_to_file(temp_file, sentence)
            audio_segment = AudioSegment.from_wav(temp_file)
            combined_audio += audio_segment + silence

            # 処理が終わった一時ファイルを削除
            os.remove(temp_file)
        except Exception as e:
            print(f"Error generating audio for sentence {idx}: {e}")
            raise e

    # 結合した音声を保存
    combined_audio.export(output_file, format="wav")
    print(f"Audio file '{output_file}' has been saved.")

# 各セクションとそのsummaryを処理
def process_article(pipe, article, base_dir, prefix="01"):
    # セクション内の summary を処理
    sentences = split_text_into_sentences(article.summary)
    audio_file = os.path.join(base_dir, f"{prefix}_summary.wav")
    generate_audio(pipe, sentences, audio_file)

    # 各セクションの処理
    for idx, section in enumerate(article.sections, start=1):
        section_prefix = f"02_{str(idx).zfill(2)}"
        section_filepath = os.path.join(base_dir, f"{section_prefix}_{sanitize_filename(section.title)}.wav")
        
        # 各セクションの summary 処理
        sentences = split_text_into_sentences(section.summary)
        summary_audio_file = os.path.join(base_dir, f"{section_prefix}_summary.wav")
        generate_audio(pipe, sentences, summary_audio_file)
        
        # 各セクションの本文を処理
        sentences = split_text_into_sentences(section.text)
        generate_audio(pipe, sentences, section_filepath)

# Wikipediaから記事を取得し音声化するクラス
class WikiAudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Wikipedia to Audio")

        self.init_gui()
        self.init_pipeline()

    # GUIの初期化
    def init_gui(self):
        self.title_label = tk.Label(self.root, text="Wikipedia Article Title:")
        self.title_label.pack()

        self.title_entry = tk.Entry(self.root, width=50)
        self.title_entry.pack()

        self.search_button = tk.Button(self.root, text="Convert Article to Audio", command=self.convert_article_to_audio)
        self.search_button.pack()

    # パイプラインの初期化
    def init_pipeline(self):
        check_cuda()
        self.pipe = init_pipeline()

    # 記事タイトルを取得し、バックグラウンドで処理を開始
    def convert_article_to_audio(self):
        article_title = self.title_entry.get().strip()
        if not article_title:
            self.show_error("Article title cannot be empty!")
            return

        # バックグラウンドで音声変換処理を実行
        threading.Thread(target=self.run_in_background, args=(article_title,)).start()

    # バックグラウンドで記事の処理を行う
    def run_in_background(self, article_title):
        article = self.fetch_wikipedia_article(article_title)
        if article:
            base_dir = os.path.join("out", sanitize_filename(article_title))
            ensure_output_directory(base_dir)
            process_article(self.pipe, article, base_dir)
        else:
            self.show_error("Article not found!")

    # Wikipediaから記事を取得
    def fetch_wikipedia_article(self, article_title):
        user_agent = f"Extensive Listening Trainer/1.0 ({CONTACT_EMAIL})"
        wiki = wikipediaapi.Wikipedia('en', headers={'User-Agent': user_agent})
        article = wiki.page(article_title)

        if article.exists():
            return article
        return None

    # エラーメッセージ表示
    def show_error(self, message):
        messagebox.showerror("Error", message)

# アプリケーションの起動
root = tk.Tk()
app = WikiAudioApp(root)
root.mainloop()
