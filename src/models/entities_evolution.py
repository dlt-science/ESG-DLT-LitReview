import os.path

from tqdm.asyncio import tqdm_asyncio
from predict_model import init_model, inference
import pandas as pd
import asyncio
from process_papers_async import load_text
import argparse


async def entities_words(predictions):
    """
    Find the entities in a document
    """

    # Get the entities from the predictions
    entities = {}
    for entity in predictions:
        label = entity["entity_group"]
        word = entity["word"]

        if label not in entities.keys():
            entities[label] = []

        entities[label].append(word)

    return entities


async def task(model, tokenizer, work_queue, semaphore, results):
    # limit the number of concurrent tasks
    async with semaphore:

        # Get the next item from the queue
        row = await work_queue.get()

        print("Loading text...")
        text = await load_text(row['source_txt_filepath'])

        if text:
            print("Doing inference...")
            predictions = inference(model, tokenizer, text)

            if predictions:
                entities = await entities_words(predictions)

                if entities:

                    # Convert the entities to a dataframe
                    df = pd.DataFrame.from_dict(entities, orient='index').transpose()

                    # Convert the pd.Series row to a pandas dataframe
                    df_row = pd.DataFrame(row).T

                    # Duplicate number of rows to match the number of entities
                    df_row = pd.concat([df_row]*len(df), ignore_index=True)

                    # Merge the two dataframes
                    df_merged = pd.concat([df_row, df], axis=1)

                    # Add the dataframe to the results
                    results.append(df_merged)

                    return results


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str,
                        default="./../../models/destilbert-cased-v2-kfold")
    parser.add_argument("--text_path", type=str,
                        default="./../../data/txts/references")

    parser.add_argument("--filtered_network", type=str,
                        default="./../../data/df_final.csv")

    parser.add_argument("--output_csv_file_path", type=str,
                        default="./../../data/network_entities_evolution.csv")

    args = parser.parse_args()

    # Create the queue of work
    work_queue = asyncio.Queue()

    # Load the model to use for inference
    model, tokenizer = init_model(args.model_dir)

    # Load the filtered network
    df_source = pd.read_csv(args.filtered_network)

    # Create a semaphore to avoid overloading the system with too many tasks loading files at once
    # Based on: https://docs.python.org/3/library/asyncio-sync.html#asyncio.Semaphore
    semaphore = asyncio.Semaphore(10)

    # Put the files to process in the queue
    files = df_source['source_txt_filepath'].tolist()

    # Put some work in the queue
    for index, row in df_source.iterrows():
        row['source_txt_filepath'] = os.path.join(args.text_path, row['source_txt_filepath'])
        await work_queue.put(row)

    # Create a list to store the results
    results = []

    # Create the tasks
    tasks = [
        task(model, tokenizer, work_queue, semaphore, results)
        for _ in range(len(files))
    ]

    # Run the tasks and show progress bar
    for t in tqdm_asyncio.as_completed(tasks):
        await t

    # Convert the results to a dataframe
    df = pd.concat(results)

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
