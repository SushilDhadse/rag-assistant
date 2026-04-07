import os
from pathlib import Path
from dotenv import load_dotenv
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

key_vault_url = os.getenv("AZURE_KEY_VAULT_URL")
print(f"Key Vault URL: {key_vault_url}")

try:
    credential = DefaultAzureCredential()
    secret_client = SecretClient(vault_url=key_vault_url, credential=credential)
    secret = secret_client.get_secret("ANTHROPIC-API-KEY")
    print(f"✅ Secret fetched successfully!")
    print(f"Secret value starts with: {secret.value[:10]}...")
except Exception as e:
    print(f"❌ Error: {e}")