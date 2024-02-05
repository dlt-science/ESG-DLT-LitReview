import argparse
import os
from predict_model import init_model, inference, load_text
import shutil


def gen_ann_files(model_dir, text_path, output_dir):
    model, tokenizer = init_model(model_dir)
    text = load_text(text_path)

    file_name = os.path.basename(text_path).replace(".txt", ".ann")
    file_path = os.path.join(output_dir, file_name)

    predictions = inference(model, tokenizer, text)
    ann_lines = []
    for i, pred in enumerate(predictions):
        # Get the annotation line
        ann_lines.append(
            f"T{str(i + 1)}" + "\t" + pred["entity_group"] + " " + str(pred["start"]) + " " + str(pred["end"]) + "\t" +
            pred["word"] + "\n")

    # Write the annotation lines to the ann file
    with open(file_path, "w") as f:
        f.writelines(ann_lines)

    # Copy the text file to the output directory with the accompanying ann file
    shutil.copy(text_path, os.path.join(output_dir, os.path.basename(text_path)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str,
                        default="./../../models/destilbert-cased/")
    # parser.add_argument("--text_path", type=str, description="Path to the text file to do inference on")
    parser.add_argument("--output_dir", type=str, default="./../../data/raw/")

    args = parser.parse_args()

    ## To label a single text file
    # gen_ann_files(args.model_dir, args.text_path, args.output_dir)

    txt_files = [f for f in os.listdir("./../../data/original") if f.endswith(".txt")]

    for text_path in txt_files:
        text_path = os.path.join("./../../data/original", text_path)
        gen_ann_files(args.model_dir, text_path, args.output_dir)
