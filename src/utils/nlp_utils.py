import re
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ── Label mapping ─────────────────────────────────────────────
# Must match exactly what was used during training
label_names = ['Account', 'Delivery', 'Other', 'Product Issue', 'Refund']

label2id = {label: idx for idx, label in enumerate(label_names)}
id2label  = {idx: label for label, idx in label2id.items()}

# ── Load model and tokenizer once at module level ─────────────
MODEL_PATH = 'src/nlp_model/ticket_classifier'

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model     = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model     = model.to(device)
model.eval()


# ── Ticket classification ─────────────────────────────────────
def predict_ticket(text: str) -> dict:
    encoding = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    input_ids      = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    probs      = torch.softmax(outputs.logits, dim=1)[0]
    pred_id    = torch.argmax(probs).item()
    pred_label = id2label[pred_id]
    confidence = probs[pred_id].item()

    return {
        'category':   pred_label,
        'confidence': round(confidence * 100, 2)
    }


# ── Entity extraction ─────────────────────────────────────────
def extract_entities(text: str) -> list:
    entities = []

    for match in re.finditer(r'ORD-\d+', text):
        entities.append({
            'label': 'ORDER_ID',
            'text':  match.group()
        })

    for match in re.finditer(r'\d{4}-\d{2}-\d{2}', text):
        entities.append({
            'label': 'DATE',
            'text':  match.group()
        })

    for match in re.finditer(r'[\w\.-]+@[\w\.-]+\.\w+', text):
        entities.append({
            'label': 'EMAIL',
            'text':  match.group()
        })

    return entities