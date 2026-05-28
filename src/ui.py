"""
Shared Rich console and CLI print helpers.

All Rich markup lives here.
"""

import json
from pathlib import Path

from rich.console import Console
from rich.json import JSON
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.theme import Theme

from src.util import root_cause

# Custom theme
custom_theme = Theme(
    {"info": "#f5f5dc", "success": "#99cc33", "warning": "#ffcc00", "danger": "#cc3300"}
)

console = Console(theme=custom_theme)


def make_progress_bar() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def print_generation_error(category: str, error: Exception) -> None:
    cause = root_cause(error)
    console.print(
        f"\n[bold danger]Generation Failed[/bold danger]  [dim]{escape(category)}[/dim]"
    )
    body = getattr(cause, "body", None)
    if body:
        console.print(JSON(json.dumps(body, indent=2)))
    else:
        message = getattr(cause, "message", None) or str(cause).splitlines()[0]
        console.print(
            f"  [danger]{type(cause).__name__}:[/danger] {escape(str(message))}"
        )


def print_batch_summary(success: int, failed: int) -> None:
    failed_generation_style = "danger" if failed > 0 else ""

    console.print(
        f"\n[success]✓ Batch complete:[/success] "
        f"{success} generated, [{failed_generation_style}]{failed} failed[/{failed_generation_style}]"
    )


def print_save_confirmation(count: int, path: Path) -> None:
    console.print(f"[bold]Saved[/bold] {count} records → [cyan]{path}[/cyan]")


def print_error(error_title: str, error_message: str, suggestion: str = "") -> None:
    content = f"\n{error_message}"
    if suggestion:
        content += f"\n\n[yellow]Suggestion: {suggestion}[/yellow]\n"
    console.print(
        Panel(
            content,
            title=f"[bold red]{error_title}[/bold red]",
            border_style="red",
            expand=False,
        )
    )
