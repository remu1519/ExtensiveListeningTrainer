import tkinter as tk
from tkinter import messagebox
import wikipediaapi
import torch
from whisperspeech.pipeline import Pipeline
import torchaudio
from dotenv import load_dotenv
import os
import re

# .envファイルから環境変数をロード
load_dotenv()

# 環境変数から連絡先を取得
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL")

# GPU（CUDA）の利用可否を確認
def check_cuda():
    if not torch.cuda.is_available():
        raise BaseException("This application requires CUDA. Please run this on a machine with a GPU.")
    else:
        print("CUDA is available. Proceeding with the pipeline.")

# WhisperSpeechのパイプライン初期化
def init_pipeline():
    return Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-tiny-en+pl.model')

# ファイル名に使用できない文字を除去する関数
def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "", filename)

# 出力ディレクトリを作成する関数
def ensure_output_directory(directory="out"):
    if not os.path.exists(directory):
        os.makedirs(directory)

# GUIの設定
class WikiAudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Wikipedia to Audio")

        # 記事名入力欄
        self.title_label = tk.Label(root, text="Wikipedia Article Title:")
        self.title_label.pack()

        self.title_entry = tk.Entry(root, width=50)
        self.title_entry.pack()

        # ボタン
        self.search_button = tk.Button(root, text="Convert Article to Audio", command=self.convert_article_to_audio)
        self.search_button.pack()

        # WhisperSpeechのパイプラインを初期化
        check_cuda()
        self.pipe = init_pipeline()
        self.article = None

    def convert_article_to_audio(self):
        # Wikipediaの記事を検索する際に、適切なユーザーエージェントを設定
        user_agent = f"Extensive Listening Trainer/1.0 ({CONTACT_EMAIL})"  # アプリ名と連絡先を含める
        
        # user_agentをキーワード引数として指定
        wiki = wikipediaapi.Wikipedia('en', headers={'User-Agent': user_agent})
        
        article_title = self.title_entry.get()
        self.article = wiki.page(article_title)

        if self.article.exists():
            self.text_to_audio(self.article.text, article_title)
        else:
            messagebox.showerror("Error", "Article not found!")

    def text_to_audio(self, text, article_title):
        # WhisperSpeechでテキストを音声に変換
        result = self.pipe.generate(text)

        # CUDAテンソルが含まれている場合はCPUに移動
        result = result.cpu()

        # 記事名をファイル名に変換して、使用できない文字を除去
        sanitized_title = sanitize_filename(article_title)
        audio_file = os.path.join("out", f"{sanitized_title}.wav")

        # 出力ディレクトリが存在しない場合は作成
        ensure_output_directory("out")

        # 音声データをWAVファイルとして保存
        torchaudio.save(audio_file, result, sample_rate=22050)

        print(f"Audio saved as {audio_file}")
        messagebox.showinfo("Success", f"Article has been converted to audio and saved as {audio_file}")

# アプリケーションの起動
root = tk.Tk()
app = WikiAudioApp(root)
root.mainloop()
