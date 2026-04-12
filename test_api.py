import asyncio
import httpx
from rich.console import Console

console = Console()

async def test_apis():
    base_url = "http://127.0.0.1:8000"
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        console.rule("[bold blue]1. Testing /health Endpoint")
        try:
            r = await client.get(f"{base_url}/health")
            console.print(f"Status: {r.status_code}")
            console.print(r.json())
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            
        console.rule("[bold blue]2. Testing /api/forecast/predict Endpoint")
        try:
            payload = {
                "sku_id": "SKU-999",
                "horizon_days": 14,
                "confidence_level": 0.95
            }
            r = await client.post(f"{base_url}/api/forecast/predict", json=payload)
            console.print(f"Status: {r.status_code}")
            console.print(r.json())
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            
        console.rule("[bold blue]3. Testing /api/forecast/chat Endpoint")
        try:
            payload = {
                "message": "What is the expected demand for winter jackets?",
                "session_id": "test_session_123"
            }
            r = await client.post(f"{base_url}/api/forecast/chat", json=payload)
            console.print(f"Status: {r.status_code}")
            console.print(r.json() if r.status_code == 200 else r.text)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

if __name__ == "__main__":
    asyncio.run(test_apis())
