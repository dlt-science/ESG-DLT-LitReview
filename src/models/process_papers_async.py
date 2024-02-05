import argparse
from predict_model import init_model, inference
import os
from tqdm.asyncio import tqdm_asyncio
import pandas as pd
import json
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


async def entities_density(text, tokenizer, predictions):
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


async def load_metadata(metadata_file):
    """
    Load the metadata file
    """

    async with aiofiles.open(metadata_file, "r") as f:
        metadata = json.loads(await f.read())

    return metadata


async def load_text(path):
    async with aiofiles.open(path, "r") as f:
        text = await f.read()
    return text


async def save_metadata(output_metadata_path, subdir_metadata, file_name, metadata):
    output_dir = os.path.join(output_metadata_path, subdir_metadata)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_metadata_file = os.path.join(output_dir, os.path.basename(file_name))

    print(f"Saving metadata for {file_name}...")
    async with aiofiles.open(output_metadata_file, "w", encoding="utf-8") as f:
        await f.write(json.dumps(metadata, indent=4))
        await f.flush()


async def task(model, tokenizer, output_metadata_path, level, work_queue, results, semaphore):

    # limit the number of concurrent tasks
    async with semaphore:

        # Get the next item from the queue
        files = await work_queue.get()

        text_file = files[0]
        metadata_file = files[1]

        print("Loading text...")
        text = await load_text(text_file)

        if text:
            print("Doing inference...")
            predictions = inference(model, tokenizer, text)

            if predictions:
                density_entities = await entities_density(text, tokenizer, predictions)

                # Update the metadata file with the density of entities
                subdir = os.path.join(level, os.path.split(os.path.split(text_file)[0])[-1])

                metadata = await load_metadata(metadata_file)
                if metadata:
                    metadata['txt_filepath'] = os.path.join(subdir,
                                                            os.path.basename(text_file))

                    metadata = metadata | density_entities

                    # Save the updated metadata file
                    await save_metadata(output_metadata_path,
                                        subdir,
                                        os.path.basename(metadata_file),
                                        metadata)

                    # Save the results to create a csv file as final report
                    results.append(metadata)


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str,
                        default="./../../models/destilbert-cased-v2-kfold")
    parser.add_argument("--text_path", type=str,
                        default="./../../data/txts/references")
    parser.add_argument("--metadata_path", type=str,
                        default="./../../data/metadata/references")
    parser.add_argument("--output_metadata_path", type=str,
                        default="./../../data/metadata/references_updated")

    parser.add_argument("--output_csv_file_path", type=str,
                        default="./../../data/predictions_references.csv")

    args = parser.parse_args()

    # Create the queue of work
    work_queue = asyncio.Queue()

    # Load the model to use for inference
    model, tokenizer = init_model(args.model_dir)

    # Initiate results
    results = []

    # Create a semaphore to avoid overloading the system with too many tasks loading files at once
    # Based on: https://docs.python.org/3/library/asyncio-sync.html#asyncio.Semaphore
    semaphore = asyncio.Semaphore(10)

    # Match the text file names to the metadata file names
    for level in ['seed_papers', 'first_level', 'second_level']:
        print(f"Processing {level}...")

        print("Getting files...")
        txt_files = get_files(os.path.join(args.text_path, level), 'txt')
        metadata_files = get_files(os.path.join(args.metadata_path, level), 'json')
        text_to_metadata = match_text_to_metadata(txt_files, metadata_files)

        # Put some work in the queue
        for text_file, metadata_file in text_to_metadata.items():
            await work_queue.put([text_file, metadata_file])

        # Create the tasks
        tasks = [
            task(model, tokenizer, args.output_metadata_path, level, work_queue, results, semaphore)
            for _ in range(len(text_to_metadata))
        ]

        # Run the tasks and show progress bar
        for t in tqdm_asyncio.as_completed(tasks):
            await t

        # Run the tasks and show progress bar
        # for t in tqdm_asyncio.as_completed(tasks):
        #     await t

        # for files in tqdm(enumerate(text_to_metadata), total=len(text_to_metadata)):
        #     tasks = [
        #         asyncio.create_task(
        #             task(model, tokenizer, args.output_metadata_path, level, work_queue, results)
        #         )
        #     ]

        # Run the tasks
        # await asyncio.gather(*tasks)
        # Using tqdm.gather to show the progress bar
        # await tqdm_asyncio.gather(*tasks)

    # Save the results to create a csv file as final report
    df = pd.json_normalize(results)

    print(f"Saving csv report results to {args.output_csv_file_path}...")
    df.to_csv(args.output_csv_file_path, index=False)


if __name__ == "__main__":
    """
    The choice of Async Programming is because the tasks are I/O bound and not CPU bound, which means that
    the tasks are waiting for something to happen (I/O) rather than using the CPU to perform other operations
    such as reading a file, writing to a file, etc.

    The advantage of Async Programming is that it allows us to do other things while waiting for the I/O to finish.

    """
    asyncio.run(main())
