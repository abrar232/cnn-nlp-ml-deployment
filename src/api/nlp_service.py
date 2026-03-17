import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.utils.nlp_utils import predict_ticket, extract_entities

# ── Load environment variables ────────────────────────────────
load_dotenv()

# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(title='Ticket Triage Service')

# ── Anthropic client ──────────────────────────────────────────
anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

# ── Request model ─────────────────────────────────────────────
class TicketRequest(BaseModel):
    text: str

# ── Endpoints ─────────────────────────────────────────────────
@app.get('/health')
def health():
    return {'status': 'ok', 'service': 'ticket-triage'}


@app.post('/triage')
def triage(request: TicketRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail='Ticket text cannot be empty')

    # Step 1 — classify the ticket
    classification = predict_ticket(request.text)

    # Step 2 — extract entities
    entities = extract_entities(request.text)

    # Step 3 — generate draft response via Anthropic
    entity_str = ''
    for e in entities:
        entity_str += f"\n  - {e['label']}: {e['text']}"
    if not entity_str:
        entity_str = '\n  - None found'

    prompt = f"""You are a customer support agent for an e-commerce company.

A customer has sent the following support ticket:
\"\"\"{request.text}\"\"\"

Ticket category: {classification['category']}
Extracted entities:{entity_str}

Write a professional, concise first response to this customer.
Rules:
- Be empathetic and polite
- Reference only the information actually present in the ticket
- Do NOT invent order details, dates or names that aren't provided
- If critical information is missing, ask for it
- Keep it under 80 words"""

    message = anthropic_client.messages.create(
        model='claude-sonnet-4-6',
        max_tokens=300,
        messages=[{'role': 'user', 'content': prompt}]
    )

    draft = message.content[0].text

    return {
        'category':      classification['category'],
        'confidence':    classification['confidence'],
        'entities':      entities,
        'draft_response': draft
    }