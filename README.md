# Conversation Ledger v0
Personal CLI for ingesting exported LLM conversations, extracting verbatim claims, clustering them, and rendering recurrence/reversal views.
Install: `pip install -r requirements.txt`
1. `python ledger.py ingest sample_export --reset`
2. `python ledger.py extract`
3. `python ledger.py cluster`
4. `python ledger.py stats`
5. `python ledger.py view`
Real export: `python ledger.py --db ledger_real.db ingest conversation_data/anthropic_export --reset` -> `python ledger.py --db ledger_real.db extract` -> `python ledger.py --db ledger_real.db cluster --backend local --threshold 0.75` -> `python ledger.py --db ledger_real.db stats` -> `python ledger.py --db ledger_real.db view --min-count 2 > view_real.txt`
Mechanical signals only. This tool maps claims, never the person. It quotes verbatim and never scores.
