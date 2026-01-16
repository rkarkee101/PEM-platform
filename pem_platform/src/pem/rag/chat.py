from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel

from pem.rag.index import RagIndex


@dataclass(frozen=True)
class ChatConfig:
    top_k: int = 5


def run_retrieval_chat(index_dir: str, *, top_k: int = 5) -> None:
    """A minimal interactive shell for RAG-style retrieval.

    This does NOT call an LLM. It retrieves relevant passages and prints them.
    In a production system, you would pass these passages as context to your preferred LLM.
    """

    console = Console()
    idx = RagIndex.load(index_dir)

    console.print(Panel.fit("PEM Retrieval Chat (type 'exit' to quit)", title="PEM"))
    while True:
        try:
            q = console.input("\n[bold]You[/bold]> ")
        except (KeyboardInterrupt, EOFError):
            console.print("\nExiting.")
            return
        if not q.strip():
            continue
        if q.strip().lower() in {"exit", "quit"}:
            console.print("Exiting.")
            return

        hits = idx.query(q, top_k=top_k)
        if not hits:
            console.print("No results.")
            continue

        for score, chunk in hits:
            header = f"{chunk.source}  (chunk {chunk.chunk_id})  score={score:.3f}"
            console.print(Panel(chunk.text, title=header))
