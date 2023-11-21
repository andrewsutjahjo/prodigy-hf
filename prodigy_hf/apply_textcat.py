import argparse
import json
from prodigy.components.stream import get_stream

from prodigy.util import log, msg
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    pipeline
)
from transformers.utils.logging import set_verbosity_error as set_transformers_verbosity_error

def write_json_to_file(file_path, data_generator):
    with open(file_path, 'w') as file:
        for item in data_generator:
            # Serialize the Python object to a JSON formatted string
            json_string = json.dumps(item)

            # Write the JSON string to the file followed by a newline character
            file.write(json_string + '\n')


def stream_model_predictions(stream, hf_pipeline, model_labels, tokenizer):
    i = 0
    for ex in stream:
        # out = hf_pipeline(
        #     tokenizer.convert_ids_to_tokens(
        #         tokenizer.encode(ex["text"],truncation=True, padding=True)
        #     )
        # )[0]
        #        score = out['score']
        try:
            out = hf_pipeline(ex["text"])[0]
        except RuntimeError:
            msg.warn(f"Failed to process example {ex['text']}.")
            continue
        ex['options'] = []
        for lab in model_labels:
            option = {"id": lab, "text": lab}
            if lab == out['label']:
                option['meta'] = out['score']
            ex['options'].append(option)
        ex['accept'] = [out['label']]
        i += 1
        if i % 100 == 0:
            msg.info(f"Processed {i} examples.")
        yield ex


def main():
    """Use transformer model to help you annotate textcat data."""
    parser = argparse.ArgumentParser(description='Description of your script.')
    parser.add_argument('-m', '--model', help='Specify the input file path', required=True)
    parser.add_argument('-o', '--output', help='Specify the output file path', required=True)
    parser.add_argument('-s', '--source', help='Specify the source file path', required=True)

    args = parser.parse_args()

    # Access the values of the arguments using args.argument_name
    model = args.model
    output = args.output
    source = args.source
    log("Run transformer on all data")
    set_transformers_verbosity_error()

    stream = get_stream(source, rehash=True, dedup=True)
    log("RECIPE: Applying tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(model)
    tfm_model = pipeline("text-classification", model=model, tokenizer=tokenizer)

    model_labels = list(tfm_model.model.config.label2id.keys())
    log(f"RECIPE: Transformer model loaded with {model_labels=}.")

    stream.apply(stream_model_predictions, hf_pipeline=tfm_model,
                 model_labels=model_labels, tokenizer=tokenizer)
    write_json_to_file(output, stream)
    msg.info("Done")


if __name__ == "__main__":
    main()
