import sys
sys.path.append("")
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")  
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
model.to(device)

sentence = "VietAI là tổ chức phi lợi nhuận với sứ mệnh ươm mầm tài năng về trí tuệ nhân tạo và xây dựng một cộng đồng các chuyên gia trong lĩnh vực trí tuệ nhân tạo đẳng cấp quốc tế tại Việt Nam."
text =  "vietnews: " + sentence + " </s>"
encoding = tokenizer(text, return_tensors="pt")
input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
outputs = model.generate(
    input_ids=input_ids, attention_mask=attention_masks,
    max_length=256,
    early_stopping=True
)
for output in outputs:
    line = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(line)