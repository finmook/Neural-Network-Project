BiLSTM:
text → word index → embedding → BiLSTM → sigmoid

Transformer:
text → subword token id → pretrained DistilBERT → classification head → softmax

Raw text
↓
AutoTokenizer
↓
input_ids + attention_mask
↓
DistilBERT
↓
Classification head
↓
logits 2 ค่า
↓
softmax / argmax
↓
non-offensive หรือ offensive