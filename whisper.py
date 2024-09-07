import tkinter as tk
from tkinter import messagebox
import wikipediaapi
import torch
import scipy.io.wavfile as wavfile  # 追加
from whisperspeech.pipeline import Pipeline
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
    return Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-small-en+pl.model')

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
            # 記事全文を取得
            full_text = self.get_full_text(self.article)
            self.text_to_audio(full_text, article_title)
        else:
            messagebox.showerror("Error", "Article not found!")

    def get_full_text(self, page):
        """
        ページ内の全セクションを再帰的に取得し、全文をテキストとして返す。
        """
        def recurse_sections(sections, text):
            for section in sections:
                text += section.title + "\n" + section.text + "\n"
                text = recurse_sections(section.sections, text)
            return text

        # ページ本体のテキスト + 全セクションのテキストを取得
        full_text = page.text + "\n"
        return recurse_sections(page.sections, full_text)

    def text_to_audio(self, text, article_title):
        # 記事名をファイル名に変換して、使用できない文字を除去
        sanitized_title = sanitize_filename(article_title)
        audio_file = os.path.join("out", f"{sanitized_title}.wav")

        self.pipe.generate_to_file(audio_file, text)

        print(f"Audio file '{audio_file}' has been saved.")
        messagebox.showinfo("Success", f"Article has been converted to audio and saved as {audio_file}")

# アプリケーションの起動
root = tk.Tk()
app = WikiAudioApp(root)
root.mainloop()
