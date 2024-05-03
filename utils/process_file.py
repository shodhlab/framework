from utils.misc import clean_text


def process_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()
        words = content.split()
        words = clean_text(words)
    return words
