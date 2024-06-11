import os
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig


OUTPUT_DIR = "Vietnam_T5_small_200"

def get_training_corpus(raw_datasets):
        return (
            raw_datasets['train'][i : i + 1000]["article"]
            for i in range(0, len(raw_datasets["train"]), 1000)
        )

def train_tokenizer(output_dir:str="Vietnam_T5_small_200", 
                    model_dir:str="Falconsai/text_summarization",
                     dataset_dir:str="Yuhthe/vietnews",
                     vocab_size:int=30_000,
                     push_to_hub:bool =False,):
    
    raw_datasets = load_dataset(dataset_dir)
    print('Data info:\n',raw_datasets)

    training_corpus = get_training_corpus(raw_datasets)

    # Initialize a tokenizer
    old_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # Customize training
    tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, vocab_size=vocab_size,
                                                        )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tokenizer.save_pretrained(output_dir)
    if push_to_hub:
        tokenizer.push_to_hub(output_dir)
    return 

if __name__ == "__main__":
    # train_tokenizer(push_to_hub=True)
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)

    # print(
    #     tokenizer.tokenize("Tôi đang ăn cơm ở nhà bà hai xã trưởng thành phố")
    # )

    # print(
    #     tokenizer.convert_ids_to_tokens(tokenizer.encode("Tôi ăn cơm"))
    # )
    # from transformers import T5Config
    # print(tokenizer.vocab_size)

    # config = T5Config.from_pretrained("google-t5/t5-small", vocab_size=tokenizer.vocab_size)

    # config.save_pretrained(OUTPUT_DIR)


