"""
Shared Rich console and CLI print helpers.

All Rich markup lives here.
"""

import json
import sys
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
from rich.rule import Rule
from rich.theme import Theme

from src.schemas import QARecord
from src.util import print_root_cause

# Custom theme
custom_theme = Theme({"success": "#99cc33", "warning": "#ffcc00", "error": "#cc3300"})

console = Console(theme=custom_theme)

# General ui utilities


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


# step 1 - generation


def print_generation_error(category: str, error: Exception) -> None:
    cause = print_root_cause(error)
    console.print(
        f"\n[bold error]Generation Failed[/bold error]  [dim]{escape(category)}[/dim]"
    )
    body = getattr(cause, "body", None)
    if body:
        console.print(JSON(json.dumps(body, indent=2)))
    else:
        message = getattr(cause, "message", None) or str(cause).splitlines()[0]
        console.print(
            f"  [error]{type(cause).__name__}:[/error] {escape(str(message))}"
        )


def print_batch_summary(success: int, failed: int) -> None:
    failed_style = "error" if failed > 0 else "default"
    console.print(
        f"\n[success]✓ Batch complete:[/success] "
        f"{success} generated, [{failed_style}]{failed} failed[/{failed_style}]"
    )


# step 3 - human labeling


def print_label_session_header(
    total_validated: int, already_labeled: int, this_session: int
) -> None:
    console.print(
        f"\n[bold]Human Labeling — Step 3[/bold]\n"
        f"  Validated: [bold]{total_validated}[/bold]  "
        f"  Already labeled: [bold]{already_labeled}[/bold]  "
        f"  This session: [bold cyan]{this_session}[/bold cyan]\n"
        f"  [dim]Progress is saved after each item.  Ctrl-C exits safely.[/dim]\n"
    )


def print_label_item(record: QARecord, idx: int, total: int) -> None:
    qa = record.record
    console.print(
        Rule(
            f"[bold] {idx}/{total} [/bold]  "
            f"[cyan]{record.category}[/cyan] [dim]›[/dim] [dim]{record.subcategory}[/dim]  "
            f"[dim]id:{record.trace_id[:8]}[/dim]"
        )
    )
    console.print(f"\n[dim]Q[/dim]  [bold]{qa.question}[/bold]\n")
    console.print(f"[dim]A[/dim]  {qa.answer}\n")

    tools_str = "  [dim]·[/dim]  ".join(qa.tools_required)
    console.print(f"[dim]Tools[/dim]   {tools_str}\n")

    console.print(
        Panel(
            f"[yellow]⚠[/yellow]  {qa.safety_info}",
            border_style="yellow",
            padding=(0, 2),
        )
    )

    steps_lines = "\n".join(
        f"  [dim]{i}.[/dim]  {s}" for i, s in enumerate(qa.steps, 1)
    )
    console.print(f"\n[bold]Steps[/bold]\n{steps_lines}\n")

    tips_lines = "\n".join(f"  [dim]·[/dim]  {t}" for t in qa.tips)
    console.print(f"[bold]Tips[/bold]\n{tips_lines}\n")


def ask_dimension(
    dim_label: str, hint: str, dim_idx: int, dim_total: int
) -> int | None:
    while True:
        try:
            raw = (
                console.input(
                    f"  [dim][{dim_idx}/{dim_total}][/dim] [bold]{dim_label}[/bold]\n"
                    f"  [dim]{hint}[/dim]\n"
                    f"  [green]p[/green]=pass  [red]f[/red]=fail  [yellow]s[/yellow]=skip  [dim]▶[/dim] "
                )
                .strip()
                .lower()
            )
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Interrupted — progress saved.[/yellow]")
            sys.exit(0)

        if raw in ("p", "1", "y"):
            return 1
        if raw in ("f", "0", "n"):
            return 0
        if raw == "s":
            return None
        console.print("  [red]Enter p (pass), f (fail), or s (skip)[/red]")


def print_dimension_result(result: int) -> None:
    console.print(f"  {'[green]✓ pass[/green]' if result else '[red]✗ fail[/red]'}\n")


def print_label_saved(
    scores: dict[str, int], overall_pass: bool, field_names: list[str]
) -> None:
    dim_row = "  ".join(
        "[green]P[/green]" if scores[fn] else "[red]F[/red]" for fn in field_names
    )
    style = "green" if overall_pass else "red"
    console.print(
        f"  Saved  {dim_row}  "
        f"[bold {style}]{'PASS' if overall_pass else 'FAIL'}[/bold {style}]\n"
    )


def print_label_session_complete(
    labeled_count: int, total_labeled: int, path: Path
) -> None:
    console.print(Rule())
    console.print(
        f"[bold green]Session complete.[/bold green]  "
        f"Labeled: [bold]{labeled_count}[/bold]  "
        f"Total on file: [bold]{total_labeled}[/bold]  "
        f"[dim]→[/dim] [cyan]{path}[/cyan]"
    )
    if total_labeled < 20:
        console.print(
            f"\n[yellow]Target is 20 labels — run again to label "
            f"{20 - total_labeled} more.[/yellow]"
        )
