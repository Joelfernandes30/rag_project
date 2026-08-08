import os
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables from .env
load_dotenv()


def test_supabase_vector_store():
    print("=" * 60)
    print("TESTING SUPABASE PGVECTOR STORE & SEARCH")
    print("=" * 60)

    url = os.getenv("SUPA_BASE_URL")
    key = os.getenv("SUPA_BASE_SCERET_KEY") or os.getenv("SUPA_BASE_API_KEY")
    table_name = "charaka_samhita_vectors"

    print(f"Supabase URL: {url}")
    print(f"Supabase Key: {'Configured' if key else 'Missing'}")
    print("-" * 60)

    if not url or not key:
        print("[ERROR]: SUPA_BASE_URL or SUPA_BASE_SCERET_KEY missing in .env")
        return False

    try:
        sb = create_client(url, key)
        print("[SUCCESS] Supabase REST Client Connected.")

        # 1. Check Table Count
        res = sb.table(table_name).select("id", count="exact").limit(1).execute()
        count = res.count if res.count is not None else len(res.data)
        print(f"[SUCCESS] Table '{table_name}' is online!")
        print(f"   Current Record Count: {count}")

        # 2. Test Upserting a Test Vector (768 dimensions)
        test_id = "test_chunk_001"
        dummy_vector = [0.01] * 768  # 768-dim test vector
        
        test_record = {
            "id": test_id,
            "content": "Test Ayurvedic remedy passage for Vata dosha balancing.",
            "section": "Test Section",
            "chapter": "Test Chapter 1",
            "pages": "1-5",
            "page_number": "1",
            "source_file": "chapters/test.md",
            "chapter_label": "Test Section - Test Chapter 1",
            "embedding": dummy_vector
        }

        print("Testing vector upsert into Supabase...")
        sb.table(table_name).upsert(test_record).execute()
        print("[SUCCESS] Test record upserted!")

        # 3. Test Vector Search via RPC match_documents
        print("Testing similarity search via 'match_documents' RPC...")
        rpc_res = sb.rpc(
            "match_documents",
            {
                "query_embedding": dummy_vector,
                "match_count": 1
            }
        ).execute()

        if rpc_res.data:
            match = rpc_res.data[0]
            print(f"[SUCCESS] RPC match_documents returned match:")
            print(f"   ID        : {match.get('id')}")
            print(f"   Content   : {match.get('content')}")
            print(f"   Similarity: {match.get('similarity')}")
        else:
            print("[NOTICE] RPC match_documents returned no results.")

        # Clean up test record
        sb.table(table_name).delete().eq("id", test_id).execute()
        print("[SUCCESS] Cleaned up test record.")
        print("=" * 60)
        return True

    except Exception as e:
        err_msg = str(e)
        print(f"[ERROR] Testing Supabase: {err_msg}")
        if "PGRST205" in err_msg or "Could not find the table" in err_msg:
            print("   HINT: Please execute schema.sql in your Supabase Dashboard SQL Editor first.")
        print("=" * 60)
        return False


if __name__ == "__main__":
    test_supabase_vector_store()
