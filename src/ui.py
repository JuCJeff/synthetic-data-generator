"""
Shared Rich console and CLI print helpers.

All Rich markup lives here.
"""

from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

console = Console()


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
    console.print(
        f"  [bold red][!] Generation failed[/bold red] [dim]({category})[/dim]"
    )
    console.print(f"  [red]{type(error).__name__}:[/red] {str(error).splitlines()[0]}")


def print_batch_summary(success: int, failed: int) -> None:
    console.print(
        f"\n[bold green]✓ Batch complete:[/bold green] "
        f"{success} generated, [red]{failed} failed[/red]"
    )


def print_save_confirmation(count: int, path: Path) -> None:
    console.print(f"[bold]Saved[/bold] {count} records → [cyan]{path}[/cyan]")
