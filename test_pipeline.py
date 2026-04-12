import asyncio
import logging
from rich.console import Console
from rich.logging import RichHandler
from orchestrator.pipeline import PipelineOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

console = Console()

async def run_pipeline():
    console.rule("[bold magenta]Starting MerchMix Automated Retraining Pipeline (Layer 10)[/bold magenta]")
    
    orchestrator = PipelineOrchestrator()
    result = await orchestrator.run_full_pipeline()
    
    console.rule("[bold green]Pipeline Execution Report[/bold green]")
    if result["status"] == "success":
        console.print("[bold green][PASS] FULL CI/CD PIPELINE SUCCEEDED[/bold green]")
    else:
        console.print(f"[bold red][FAIL] PIPELINE FAILED: {result.get('error')}[/bold red]")
        
    console.print(result)

if __name__ == "__main__":
    asyncio.run(run_pipeline())
