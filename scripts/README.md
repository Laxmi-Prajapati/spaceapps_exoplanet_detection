Usage: set_gemini_key.ps1

Run the script from PowerShell to persistently set the GEMINI_API_KEY for your user and optionally create a `.env` file in the project root.

Examples:

# Persist key for user only (no .env)
./scripts/set_gemini_key.ps1

# Persist key and create/update .env in project root
./scripts/set_gemini_key.ps1 -WriteDotEnv

After running, open a new PowerShell session or run the suggested in-place command to load the variable into the current shell:

$env:GEMINI_API_KEY = [Environment]::GetEnvironmentVariable('GEMINI_API_KEY','User')
