"""Terminal entrypoint for deterministic support triage pipeline."""

from __future__ import annotations

from pathlib import Path

from agent import SupportTriageAgent
from corpus_loader import load_corpus_chunks
from retriever import DeterministicRetriever
from utils import load_env, read_ticket_csv, repo_root_from_code_dir, write_output_csv


def main() -> None:
    """Run triage over support_tickets/support_tickets.csv and write output.csv."""
    load_env()
    root = repo_root_from_code_dir(__file__)
    input_csv = root / "support_tickets" / "support_tickets.csv"
    output_csv = root / "support_tickets" / "output.csv"
    data_root = root / "data"

    tickets = read_ticket_csv(input_csv)
    chunks = load_corpus_chunks(data_root)
    retriever = DeterministicRetriever(chunks)
    agent = SupportTriageAgent(retriever)

    predictions = [agent.process_ticket(ticket) for ticket in tickets]
    write_output_csv(output_csv, predictions)

    print(f"Processed {len(tickets)} tickets.")
    print(f"Wrote predictions to: {output_csv.as_posix()}")


if __name__ == "__main__":
    main()
