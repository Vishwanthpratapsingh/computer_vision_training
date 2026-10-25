# Week 4: Document Understanding (Invoice Extraction)

Turn invoices into structured JSON two ways, a self-hosted OCR/layout pipeline and a managed cloud service, and compare them on a held-out set.

> TEMPLATE: replace every `___` and every line in (parentheses) with your own results and words
> after running `week4_deliverable.ipynb`. Delete this note when done.

## Deliverable
- `week4_deliverable.ipynb`: both pipelines, the held-out evaluation, the comparison matrix and the write-up (saved with outputs)

## Common JSON schema
```json
{
  "doc_id": "t000",
  "source": "tesseract+layout-rules",
  "fields": {"invoice_number": "...", "invoice_date": "...", "vendor_name": "...", "customer_name": "...",
             "subtotal": 0.0, "tax": 0.0, "total": 0.0},
  "line_items": [{"description": "...", "quantity": 0.0, "unit_price": 0.0, "amount": 0.0}],
  "meta": {"latency_s": 0.0, "pages": 1}
}
```
Both pipelines are validated against this JSON Schema.

## Approaches
| | Self-hosted | Managed |
|---|---|---|
| Engine | Tesseract OCR, word boxes | ___ (Azure Document Intelligence `prebuilt-invoice` / AWS Textract `AnalyzeExpense`) |
| Preprocessing | ___ (denoise, deskew, threshold, or none) | none (service handles it) |
| Fields | regex rules + position split for Seller/Client | prebuilt model |
| Tables | header-anchored column assignment, merge wrapped lines | prebuilt model |

## Held-out set
- Source: ___ (synthetic invoices dataset, or your own labelled invoices)
- Size: ___ clean invoices, ___ also run as degraded scans (rotation, blur, noise, JPEG compression)
- Dev set (used for tuning the rules): ___ documents; the held-out set was not used for tuning

## Comparison matrix
(Paste the table printed in section 10 of the notebook.)

| Criterion | Self-hosted | Managed |
|---|---|---|
| Valid JSON | ___ | ___ |
| Field accuracy (clean) | ___ | ___ |
| Field accuracy (degraded) | ___ | ___ |
| Table cell F1 (clean) | ___ | ___ |
| Table cell F1 (degraded) | ___ | ___ |
| Median latency (s/page) | ___ | ___ |
| Cost per 1,000 pages (USD) | ___ | ___ |
| Data leaves your machine | no | yes |

## Trade-offs and where each approach fails
- Accuracy and tables: ___
- Latency and cost (with what the estimate leaves out): ___
- Data privacy and residency: ___
- Self-hosted failures (example): ___
- Managed failures (example): ___

## Recommendation
- Use the managed service when: ___ (back with a number)
- Use the self-hosted pipeline when: ___ (back with a number)
- Caveat: small held-out set, so differences of a few points may be noise

## Success gate
- [ ] Both approaches produce valid structured JSON on unseen invoices
- [ ] The comparison matrix supports a defensible recommendation backed by numbers

## What this sets up
The common schema and held-out set are the baseline for the VLM comparison (Module 5) and the RAG ingestion layer (Module 6).

## Cost and privacy notes
- Cloud pages sent: ___ (capped at `MAX_CLOUD_PAGES`; results cached so nothing is paid for twice)
- No API keys or real customer documents are committed to this repo

## How to reproduce
1. Open `week4_deliverable.ipynb` in Google Colab (T4 GPU for EasyOCR and Donut).
2. Add the Colab secrets for your managed service (see section 2 of the notebook).
3. Run all cells.
