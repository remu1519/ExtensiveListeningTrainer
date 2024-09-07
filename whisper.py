import tkinter as tk
from tkinter import messagebox
import wikipediaapi
import re
from whisperspeech.pipeline import Pipeline
from dotenv import load_dotenv
import os
from pydub import AudioSegment
import torch

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
def ensure_output_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# テキストを一文ごとに分割する関数（改行で分割、!?は除外）
def split_text_into_sentences(text):
    sentences = re.split(r'(?:\n|(?<=[.!?]))+', text.strip())  # 改行または文末記号で分割
    return [sentence for sentence in sentences if sentence]

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
        article_title = self.title_entry.get().strip()
        if not article_title:
            messagebox.showerror("Error", "Article title cannot be empty!")
            return

        # Wikipediaの記事を検索する際に、適切なユーザーエージェントを設定
        user_agent = f"Extensive Listening Trainer/1.0 ({CONTACT_EMAIL})"
        wiki = wikipediaapi.Wikipedia('en', headers={'User-Agent': user_agent})
        
        self.article = wiki.page(article_title)

        if self.article.exists():
            # 概要を最初に処理
            self.process_summary(self.article.summary, article_title)
            
            # セクションごとに音声ファイルを作成
            self.process_sections(self.article.sections, article_title)
        else:
            messagebox.showerror("Error", "Article not found!")

    def process_summary(self, summary, article_title):
        # 概要（イントロダクション）の処理
        sanitized_title = sanitize_filename(article_title)
        base_dir = os.path.join("out", sanitized_title)
        ensure_output_directory(base_dir)

        # 概要を一文ごとに分割
        sentences = split_text_into_sentences(summary)
        audio_file = os.path.join(base_dir, "01_summary.wav")  # プレフィックス「01_」を追加

        # 概要の音声化
        self.text_to_audio(sentences, audio_file)

    def process_sections(self, sections, article_title):
        # 出力ディレクトリを作成
        sanitized_title = sanitize_filename(article_title)
        base_dir = os.path.join("out", sanitized_title)
        ensure_output_directory(base_dir)

        for idx, section in enumerate(sections, start=2):  # 概要は「01_」なので、セクションは「02_」から始める
            section_title = sanitize_filename(section.title)
            section_dir = os.path.join(base_dir, f"{str(idx).zfill(2)}_{section_title}.wav")  # プレフィックスを追加

            # セクションのテキストを一文ごとに分割
            sentences = split_text_into_sentences(section.text)

            # セクション内の全センテンスを結合して1つの音声ファイルにする
            self.text_to_audio(sentences, section_dir)

    def text_to_audio(self, sentences, output_file):
        combined = AudioSegment.empty()
        silence = AudioSegment.silent(duration=400)  # 0.4秒の無音を作成

        for idx, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            temp_file = os.path.join("out", "temp", f"temp_{idx}.wav")
            ensure_output_directory(os.path.dirname(temp_file))

            try:
                # 各文を音声化して一時ファイルに保存
                self.pipe.generate_to_file(temp_file, sentence)
                audio_segment = AudioSegment.from_wav(temp_file)
                combined += audio_segment + silence  # 各センテンスを結合し、0.2秒の無音を追加

                # 処理が終わった一時ファイルを削除
                os.remove(temp_file)
            except Exception as e:
                print(f"Error generating audio for sentence {idx}: {e}")
                messagebox.showerror("Error", f"Error generating audio for sentence {idx}: {e}")

        # 結合した音声を保存
        combined.export(output_file, format="wav")
        print(f"Audio file '{output_file}' has been saved.")

# アプリケーションの起動
root = tk.Tk()
app = WikiAudioApp(root)
root.mainloop()
