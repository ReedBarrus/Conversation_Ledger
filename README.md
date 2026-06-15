# Conversation Ledger v0
Personal CLI for ingesting exported LLM conversations, extracting verbatim claims, clustering them, and rendering recurrence/reversal views.
Install: `pip install sentence-transformers`
1. `python ledger.py ingest sample_export`
2. `python ledger.py extract`
3. `python ledger.py cluster`
4. `python ledger.py view`
Mechanical signals only. This tool maps claims, never the person. It quotes verbatim and never scores.
