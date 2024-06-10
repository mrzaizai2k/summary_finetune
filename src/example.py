import sys
sys.path.append("")
import torch
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from src.Utils.utils import read_text_from_file

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model_dir="vietnamese_mt5_summary_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)  
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, 
                                              torch_dtype=torch.bfloat16,).to(device)
config = AutoConfig.from_pretrained(model_dir)
print("config", config)
print("generation_config", model.generation_config)


file_path = "data/sample.txt"
text = read_text_from_file(file_path)
txt_list = 10*[text] 

start_time = time.time()
encoding = tokenizer(txt_list, return_tensors="pt")
input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

outputs = model.generate(
    input_ids=input_ids, attention_mask=attention_masks,
    max_length=256,
    # num_beams=4,
    # no_repeat_ngram_size=3,
    # early_stopping=True,
    # length_penalty= 2.0,
    do_sample=True,
    # top_p=0.8,
    penalty_alpha = 0.6,
    top_k=4,
    # temperature=0.5,
)
end_time = time.time()
processing_time = end_time - start_time
print(f"Processing time: {processing_time:.4f} seconds")

for output in outputs:
    line = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(line)





















































