# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI — text chat, single question, system info, RAG management."""

import time
from typing import Optional
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
import typer

from app.config import Config
from app.audit import AuditLogger
from app.llm import LLM
from app.rag import KnowledgeBase, RAGRetriever
from app.monitor import get_system_stats, format_stats

console = Console()
app = typer.Typer(name="assistant", help="Reachy Mini Jetson Assistant", add_completion=False)


def _load_llm(config: Config) -> LLM:
    audit = AuditLogger(
        config.runtime.audit_log_path,
        max_bytes=config.runtime.audit_max_bytes,
        backup_count=config.runtime.audit_backup_count,
    )
    llm = LLM(
        model=config.llm.model, base_url=config.llm.base_url,
        backend=config.llm.backend, max_tokens=config.llm.max_tokens,
        temperature=config.llm.temperature, timeout=config.llm.timeout,
        system_prompt=config.llm.system_prompt,
        guardrail_config=config.guardrails,
        audit_logger=audit,
    )
    if not llm.load():
        console.print("[red]LLM failed to connect[/red]")
        raise typer.Exit(1)
    return llm


def _load_rag(config: Config) -> Optional[RAGRetriever]:
    if not config.rag.enabled:
        return None
    try:
        kb = KnowledgeBase(
            persist_dir=config.rag.persist_dir,
            embedding_backend=config.rag.embedding_backend,
            embedding_model=config.rag.embedding_model,
            embedding_base_url=config.rag.embedding_base_url,
            chunk_size=config.rag.chunk_size,
            chunk_overlap=config.rag.chunk_overlap,
        )
        count, rebuilt = kb.sync_directory(config.rag.knowledge_dir)
        if rebuilt:
            console.print(f"  [cyan]RAG rebuilt ({count} chunks)[/cyan]")
        return RAGRetriever(kb, config.rag.n_results, config.rag.min_relevance)
    except Exception as e:
        console.print(f"[yellow]RAG unavailable: {e}[/yellow]")
        return None


def _stream(llm: LLM, rag: Optional[RAGRetriever], text: str, system_prompt: str):
    """Stream LLM response, print tokens. Returns (full_text, tps, ttft, tokens)."""
    prompt = rag.augment_query(text) if rag else text
    ttft = None
    t0 = time.perf_counter()
    full = ""
    meta_final = {}
    for chunk_data in llm.generate_stream(prompt=prompt, system_prompt=system_prompt):
        content, meta = chunk_data if isinstance(chunk_data, tuple) else (chunk_data, {})
        if meta.get("done"):
            meta_final = meta
        if content:
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            console.print(content, end="", highlight=False)
            full += content
    elapsed = time.perf_counter() - t0
    tokens = meta_final.get("eval_count", 0) or max(1, len(full) // 4)
    tps = tokens / elapsed if elapsed > 0 else 0
    return full, tps, ttft or 0, tokens


@app.command()
def chat(
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """Interactive text chat."""
    config = Config.load(config_path)
    if model:
        config.llm.model = model

    with console.status("[bold]Loading...[/bold]"):
        llm = _load_llm(config)
        rag = _load_rag(config)

    label = "llama.cpp" if config.llm.backend == "openai" else "Ollama"
    console.print(f"[green]✓ LLM ({label}: {llm.model})[/green]")
    if rag:
        console.print(f"[green]✓ RAG ({rag.kb.count()} chunks)[/green]")
    console.print("\n[dim]'quit' to exit, 'clear' to reset, 'stats' for system info[/dim]\n")

    try:
        while True:
            try:
                text = Prompt.ask("[cyan]You[/cyan]")
                if not text.strip():
                    continue
                if text.strip().lower() == "quit":
                    break
                if text.strip().lower() == "clear":
                    console.print("[dim]History cleared[/dim]")
                    continue
                if text.strip().lower() == "stats":
                    s = get_system_stats()
                    console.print(f"  {format_stats(s)}")
                    continue
                console.print("[magenta]Assistant[/magenta]: ", end="")
                _, tps, ttft, tok = _stream(llm, rag, text, config.llm.system_prompt)
                console.print()
                console.print(f"[dim]  TTFT: {ttft:.0f}ms | {tps:.1f} tok/s | {tok} tokens | {format_stats(get_system_stats())}[/dim]")
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted[/yellow]")
                continue
    finally:
        llm.unload()
        console.print("[yellow]Goodbye![/yellow]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """Ask a single question."""
    config = Config.load(config_path)
    if model:
        config.llm.model = model
    with console.status("[bold]Loading...[/bold]"):
        llm = _load_llm(config)
        rag = _load_rag(config)
    console.print(f"\n[cyan]Q:[/cyan] {question}\n[magenta]A:[/magenta] ", end="")
    _, tps, ttft, tok = _stream(llm, rag, question, config.llm.system_prompt)
    console.print(f"\n\n[dim]TTFT: {ttft:.0f}ms | {tps:.1f} tok/s | {tok} tokens[/dim]")
    llm.unload()


@app.command()
def info():
    """Show system info and dependency status."""
    console.print("\n[bold cyan]System Info[/bold cyan]\n")
    s = get_system_stats()
    console.print(f"  {format_stats(s)}\n")
    config = Config.load()
    console.print(
        f"  Platform: [cyan]{config.platform.display_name}[/cyan] "
        f"([dim]{config.platform.profile}, {config.platform.implementation_status}[/dim])\n"
    )
    try:
        import httpx
        with httpx.Client(timeout=5.0) as c:
            if config.llm.backend == "openai":
                r = c.get(f"{config.llm.base_url.rstrip('/')}/v1/models")
                if r.status_code == 200:
                    models = [m.get("id", "") for m in r.json().get("data", [])]
                    console.print(f"[green]✓ LLM[/green]: {', '.join(models[:3])}")
            else:
                r = c.get(f"{config.llm.base_url.rstrip('/')}/api/tags")
                if r.status_code == 200:
                    names = [m.get("name", "") for m in r.json().get("models", [])]
                    console.print(f"[green]✓ Ollama[/green]: {', '.join(names[:3])}")
    except Exception:
        console.print(f"[red]✗ LLM not running at {config.llm.base_url}[/red]")
    for lib, name in [("faster_whisper", "faster-whisper"), ("kokoro_onnx", "kokoro-onnx"), ("chromadb", "ChromaDB")]:
        try:
            __import__(lib)
            console.print(f"[green]✓ {name}[/green]")
        except ImportError:
            console.print(f"[yellow]⚠ {name} not installed[/yellow]")


@app.command()
def rag_add(
    path: str = typer.Argument(..., help="File or directory to add"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """Add documents to RAG knowledge base."""
    config = Config.load(config_path)
    kb = KnowledgeBase(
        persist_dir=config.rag.persist_dir, embedding_backend=config.rag.embedding_backend,
        embedding_model=config.rag.embedding_model, embedding_base_url=config.rag.embedding_base_url,
        chunk_size=config.rag.chunk_size, chunk_overlap=config.rag.chunk_overlap,
    )
    target = Path(path)
    if target.is_file():
        n = kb.add_file(str(target))
        console.print(f"[green]✓ Added {target.name}[/green] ({n} chunks)")
    elif target.is_dir():
        n = kb.add_directory(str(target))
        console.print(f"[green]✓ Added {n} chunks[/green]")
    else:
        console.print(f"[red]Not found: {path}[/red]")
        return
    console.print(f"Total: [cyan]{kb.count()} chunks[/cyan]")


@app.command()
def rag_status(config_path: Optional[str] = typer.Option(None, "--config", "-c")):
    """Show RAG status."""
    config = Config.load(config_path)
    kb = KnowledgeBase(
        persist_dir=config.rag.persist_dir, embedding_backend=config.rag.embedding_backend,
        embedding_model=config.rag.embedding_model, embedding_base_url=config.rag.embedding_base_url,
    )
    console.print(f"  Embedding: [cyan]{config.rag.embedding_backend}[/cyan] ({config.rag.embedding_model})")
    console.print(f"  Chunks: [green]{kb.count()}[/green]")
    console.print(f"  Enabled: {'[green]yes[/green]' if config.rag.enabled else '[yellow]no[/yellow]'}")


@app.command()
def rag_search(
    query: str = typer.Argument(...),
    n: int = typer.Option(3, "--results", "-n"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """Search the knowledge base."""
    config = Config.load(config_path)
    kb = KnowledgeBase(
        persist_dir=config.rag.persist_dir, embedding_backend=config.rag.embedding_backend,
        embedding_model=config.rag.embedding_model, embedding_base_url=config.rag.embedding_base_url,
    )
    results = kb.search(query, n_results=n)
    if not results:
        console.print("[yellow]No results[/yellow]")
        return
    for i, r in enumerate(results, 1):
        src = r.get("metadata", {}).get("filename", "?")
        console.print(f"[cyan][{i}][/cyan] {src} [dim]({1 - r.get('distance', 0) / 2:.0%})[/dim]")
        console.print(f"    {r.get('content', '')[:200]}\n")


@app.command()
def rag_clear(
    config_path: Optional[str] = typer.Option(None, "--config", "-c"),
    force: bool = typer.Option(False, "--force", "-f"),
):
    """Clear the knowledge base."""
    config = Config.load(config_path)
    if not force and not typer.confirm("Clear all documents?"):
        return
    kb = KnowledgeBase(
        persist_dir=config.rag.persist_dir, embedding_backend=config.rag.embedding_backend,
        embedding_model=config.rag.embedding_model, embedding_base_url=config.rag.embedding_base_url,
    )
    kb.clear()
    console.print("[green]✓ Cleared[/green]")


def main():
    app()


if __name__ == "__main__":
    main()
