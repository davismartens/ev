from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def step(msg: str):
    console.print(f"[bold cyan]›[/] {msg}")

def substep(msg: str):
    console.print(f"   [dim]{msg}[/dim]")

def success(msg: str):
    console.print(f"[bold green]✓[/] {msg}")

def fail(msg: str):
    console.print(f"[bold red]✗[/] {msg}")

def warn(msg: str):
    console.print(f"[yellow]![/] {msg}")

def spinner():
    return Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[progress.description]{task.description}", style="dim"),
        transient=True,
    )

def build_summary_table(summary: dict) -> Table:
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Case", width=20)
    table.add_column("Criteria", width=20)
    table.add_column("Passed", width=8)

    for case in summary["cases"]:
        case_name = case["case_name"]
        for i, obj in enumerate(case["objectives"]):
            crit = list(obj.keys())[0]
            passed = "[green]✓[/]" if obj[crit] else "[red]✗[/]"
            if i == 0:
                table.add_row(case_name, crit, passed)
            else:
                table.add_row("", crit, passed)

    return table
