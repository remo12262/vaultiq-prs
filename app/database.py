import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError(f"Mancano le variabili: URL={SUPABASE_URL[:20] if SUPABASE_URL else 'VUOTA'}")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

async def init_db():
    print(f"Connesso a Supabase: {SUPABASE_URL}")
