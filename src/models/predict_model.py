import argparse
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


def init_model(model_dir):
    # Load the model
    model = AutoModelForTokenClassification.from_pretrained(model_dir)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    return model, tokenizer


def inference(model, tokenizer, text):
    ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="first")

    # Tokenize the text into sequences of 512 tokens
    input_ids = tokenizer.encode(text, truncation=False)

    # Split the sequences into chunks of 512 tokens
    chunks = [input_ids[i:i + 512] for i in range(0, len(input_ids), 512)]

    # Perform inference on each chunk
    predictions = [ner(tokenizer.decode(chunk)) for chunk in chunks]

    # Flatten the list of predictions
    predictions = [item for sublist in predictions for item in sublist]

    return predictions


def load_text(path):
    with open(path, "r") as f:
        text = f.read()
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str,
                        default="./../models/destilbert-cased-v2-kfold/",
                        description="Path to the trained model directory")
    parser.add_argument("--text_path", type=str, description="Path to the text file to do inference on")

    args = parser.parse_args()

    model, tokenizer = init_model(args.model_dir)
    predictions = inference(model, tokenizer, args.text)
    print(predictions)
