import asyncio
import httpx
from rich.console import Console
from rich.pretty import pprint

console = Console()

async def run_e2e_tests():
    base_url = "http://127.0.0.1:8000"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        console.rule("[bold cyan]1. Testing Root (Health) Endpoint [/bold cyan]")
        try:
            r = await client.get(f"{base_url}/")
            console.print(f"Status: {r.status_code}")
            pprint(r.json() if r.status_code == 200 else r.text)
        except Exception as e:
            console.print(f"[bold red]Connection Error: {repr(e)}[/bold red]")
            console.print("Make sure your uvicorn server is running!")
            return

        console.rule("[bold cyan]2. Testing Forecast Predict Endpoint [/bold cyan]")
        try:
            payload = {
                "sku_id": "MENS-TEE-BLK-M",
                "horizon_days": 14,
                "confidence_level": 0.95
            }
            console.print("Sending payload:", payload)
            r = await client.post(f"{base_url}/api/forecast/predict", json=payload)
            console.print(f"Status: {r.status_code}")
            pprint(r.json() if r.status_code == 200 else r.text)
        except Exception as e:
            console.print(f"[bold red]API Error: {repr(e)}[/bold red]")
            
        console.rule("[bold cyan]3. Testing Interactive Agent Chat Endpoint [/bold cyan]")
        try:
            payload = {
                "message": "We have a promotion coming up for winter jackets. Can you check our current stock and tell me if we need to order more?",
                "session_id": "e2e_test_user123"
            }
            console.print("Sending payload:", payload)
            r = await client.post(f"{base_url}/api/forecast/chat", json=payload)
            console.print(f"Status: {r.status_code}")
            pprint(r.json() if r.status_code == 200 else r.text)
        except Exception as e:
            console.print(f"[bold red]API Error: {repr(e)}[/bold red]")
            
if __name__ == "__main__":
    console.print("[bold green]Starting MerchMix E2E API Diagnostics...[/bold green]")
    asyncio.run(run_e2e_tests())
