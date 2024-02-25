import argparse

import requests
import os
import ftfy

# Download the Apache Tika .jar file from https://tika.apache.org/download.html
# os.environ["TIKA_SERVER_JAR"] = "https://repo1.maven.org/maven2/org/apache/tika/tika-server-standard/2.8.0/tika-server-standard-2.8.0.jar"
os.environ[
    "TIKA_SERVER_JAR"] = "https://repo1.maven.org/maven2/org/apache/tika/tika-server-standard/3.0.0-BETA/tika-server-standard-3.0.0-BETA.jar"
# os.environ["TIKA_SERVER_ENDPOINT"] = "tika-server-standard-2.8.0.jar"

import tika

tika.initVM()

from tika import parser as tika_parser


def remove_illegal_symbols_filename(publication_name):
    # Remove symbols from the paper name not allowed in file names
    for symbol in [" ", ":", "/", "\\", "?", "*", "<", ">", "|", "\"", "\'"]:
        publication_name = publication_name.replace(symbol, "_")

    return publication_name


def parse_from_pdf_url(paper_name, pdf_url):
    # Add a request timeout in seconds
    response = requests.get(pdf_url, timeout=10)
    file_data = tika_parser.from_buffer(response.content, requestOptions={'timeout': 120})

    # Save the PDF as a file
    with open(f"{paper_name}.pdf", 'wb') as f:
        f.write(response.content)

    return file_data


def parse_from_pdf_local(local_file_path):
    # Convert PDF to text with Apache Tika
    file_data = tika_parser.from_file(local_file_path)

    return file_data


def fix_encoding_issues(text):
    # Fix any unicode errors
    text = ftfy.fix_text(text)
    text = ftfy.fix_encoding(text)
    text = text.strip()

    return text


def save_parsed_txt(paper_name, text, data_dir):
    file_path = os.path.join(data_dir, f"{paper_name}.txt")

    print(f"Saving the parsed text as a .txt file in {file_path}...")

    # Save the text content as a .txt file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()

    # URL of the PDF
    # E.g. "https://hedera.com/hh_whitepaper_v2.1-20200815.pdf"
    args_parser.add_argument(
        "--pdf_url", type=str, default=None
    )

    args_parser.add_argument(
        "--pdf_local_path", type=str, default=None
    )

    # E.g. "Hedera Whitepaper"
    args_parser.add_argument(
        "--publication_name", type=str, default=None
    )
    args_parser.add_argument(
        "--data_dir", type=str, default="./../../../data/original"
    )

    args = args_parser.parse_args()

    args.publication_name = remove_illegal_symbols_filename(args.publication_name)

    if args.pdf_url:
        print(f"Downloading and parsing the PDF from the URL: {args.pdf_url}")
        file_data = parse_from_pdf_url(args.publication_name, args.pdf_url)

    else:
        if args.pdf_local_path is None:
            raise ValueError("Please provide either a PDF URL or a local file path")

        print(f"Parsing the PDF from the local file: {args.pdf_local_path}")
        file_data = parse_from_pdf_local(args.pdf_local_path)

    # Get the text content from the parsed file_data
    print("Extracting the text from the parsed PDF...")
    text = file_data["content"]

    # Fix any unicode and other encoding errors
    print("Fixing any encoding issues...")
    text = fix_encoding_issues(text)

    # Save the parsed text as a .txt file
    save_parsed_txt(args.publication_name, text, args.data_dir)
