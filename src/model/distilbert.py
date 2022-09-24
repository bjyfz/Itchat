from transformers import AutoModelForMaskedLM

roberta = AutoModelForMaskedLM.from_pretrained("roberta-large")

print(roberta)