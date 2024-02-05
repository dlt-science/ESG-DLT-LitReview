import argparse
from predict_model import init_model, inference, load_text
import os
from tqdm import tqdm
import pandas as pd
import json
import io
import aiofiles
import asyncio


def get_files(files_path, extension):
    """
    Get all the files in a directory and its subdirectories
    with a specific extension recursively
    """

    files = []
    for dirpath, dirnames, filenames in os.walk(files_path):
        for filename in filenames:
            if filename.endswith(f".{extension}"):
                files.append(os.path.join(dirpath, filename))

    return files


def entities_density(text, tokenizer, predictions):
    """
    Calculate the density of entities in a document
    """

    tokenized_text = tokenizer.tokenize(text)
    tokens_count = len(tokenized_text)

    entities = [prediction['entity_group'] for prediction in predictions]
    unique_entities = set(entities)
    entities_count = len(entities)
    entities_density = entities_count / tokens_count

    entity_results = {}
    output = {}
    for entity in unique_entities:
        entity_results[entity] = {
            "density": entities.count(entity) / tokens_count,
            "total_entity": entities.count(entity)
        }

    output["named_entities"] = entity_results
    output["total_tokens_doc"] = tokens_count
    output["total_entities_doc"] = entities_count
    output["entities_density_doc"] = entities_density

    return output


def match_text_to_metadata(text_files, metadata_files):
    """
    Match the text file names to the metadata file names
    """

    text_files_set = set(text_files)
    metadata_files_set = set(metadata_files)

    metadata_files_hash_set = {os.path.splitext(os.path.basename(file_path))[0]: file_path for file_path in
                               metadata_files_set}

    text_to_metadata = {}
    for text_file in text_files_set:
        text_file_name = os.path.splitext(os.path.basename(text_file))[0]
        if text_file_name in set(metadata_files_hash_set.keys()):
            text_to_metadata[text_file] = metadata_files_hash_set[text_file_name]

    return text_to_metadata


def load_metadata(metadata_file):
    """
    Load the metadata file
    """

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    return metadata


def save_metadata(output_metadata_path, subdir_metadata, metadata):

    output_dir = os.path.join(output_metadata_path, subdir_metadata)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_metadata_file = os.path.join(output_dir, os.path.basename(metadata_file))

    print(f"Saving metadata for {metadata_file}...")
    with open(output_metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str,
                        default="")
    parser.add_argument("--text_path", type=str,
                        default="")
    parser.add_argument("--metadata_path", type=str,
                        default="")
    parser.add_argument("--output_metadata_path", type=str,
                        default="")

    parser.add_argument("--output_csv_file_path", type=str,
                        default="")

    args = parser.parse_args()

    # Load the model to use for inference
    model, tokenizer = init_model(args.model_dir)

    # Match the text file names to the metadata file names
    for level in ['first_level', 'second_level']:
        print(f"Processing {level}...")

        print("Getting files...")
        txt_files = get_files(os.path.join(args.text_path, level), 'txt')
        metadata_files = get_files(os.path.join(args.metadata_path, level), 'json')
        text_to_metadata = match_text_to_metadata(txt_files, metadata_files)

        results = []

        print("Doing inference...")
        for text_file, metadata_file in tqdm(text_to_metadata.items(), total=len(text_to_metadata)):

            text = load_text(text_file)
            metadata = load_metadata(metadata_file)

            predictions = inference(model, tokenizer, text)

            if predictions:
                density_entities = entities_density(text, tokenizer, predictions)

                # Update the metadata file with the density of entities
                subdir = os.path.join(level, os.path.split(os.path.split(text_file)[0])[-1])
                metadata['txt_filepath'] = os.path.join(subdir,
                                                        os.path.basename(text_file))

                metadata = metadata | density_entities

                # Save the updated metadata file
                save_metadata(args.output_metadata_path, subdir, metadata)

                # Save the results to create a csv file as final report
                results.append(metadata)

        # Save the results to create a csv file as final report
        df = pd.json_normalize(results)

        print(f"Saving csv report results to {args.output_csv_file_path}...")
        df.to_csv(args.output_csv_file_path, index=False)
