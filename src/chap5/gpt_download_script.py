import urllib.request

GPT_URL = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch05/"
    "01_main-chapter-code/gpt_download.py"
    )


if __name__ == '__main__':
    filename = GPT_URL.split('/')[-1]
    urllib.request.urlretrieve(GPT_URL, filename)