# #
# # Script: set_gemini_key.ps1
# # Purpose: Persistently set GEMINI_API_KEY for the current user and optionally create a project .env file.
# # Usage: Run in PowerShell (run as normal user):
# #     .\scripts\set_gemini_key.ps1
# #

# Param(
#     [switch]$WriteDotEnv  # if supplied, also write a .env file in project root
# )

# # Secure prompt for secret
# # Write-Host "Enter your GEMINI API key (input hidden):"
# # $secureString = Read-Host -AsSecureString
# # $plain = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureString))

# $plain = Read-Host "Enter your GEMINI API key (visible)"

# # Persist for current user
# setx GEMINI_API_KEY "$plain" | Out-Null
# Write-Host "GEMINI_API_KEY has been set for the current user. Open a new shell to see it." -ForegroundColor Green

# if ($WriteDotEnv) {
#     $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
#     $projectRoot = Resolve-Path "$scriptDir\.." | Select-Object -ExpandProperty Path
#     $envPath = Join-Path $projectRoot ".env"
#     if (Test-Path $envPath) {
#         Write-Host ".env already exists at $envPath - updating it (old GEMINI_API_KEY will be replaced)" -ForegroundColor Yellow
#         (Get-Content $envPath) | Where-Object {$_ -notmatch '^GEMINI_API_KEY='} | Set-Content $envPath
#     }
#     "GEMINI_API_KEY=$plain" | Add-Content $envPath
#     Write-Host ".env created/updated at $envPath" -ForegroundColor Green

#     # Add .env to .gitignore if not present
#     $gitignore = Join-Path $projectRoot ".gitignore"
#     if (Test-Path $gitignore) {
#         $gitignoreContent = Get-Content $gitignore -ErrorAction SilentlyContinue
#         if ($gitignoreContent -notcontains ".env") {
#             Add-Content -Path $gitignore -Value ".env"
#             Write-Host ".env added to .gitignore" -ForegroundColor Green
#         }
#     } else {
#         ".env" | Out-File -FilePath $gitignore -Encoding utf8
#         Write-Host "Created .gitignore and added .env" -ForegroundColor Green
#     }
# }

# # Clear the plain secret from memory
# Remove-Variable -Name plain -ErrorAction SilentlyContinue

# Write-Host "Done. To use the variable in existing shells, either open a new terminal or run:`"`n$env:GEMINI_API_KEY = [Environment]::GetEnvironmentVariable('GEMINI_API_KEY','User')`"`n" -ForegroundColor Cyan


import os
import sys
import argparse

# --- 1. Argument Parsing ---
# This mirrors the Param/switch functionality of the PowerShell script
parser = argparse.ArgumentParser(description="Persistently set GEMINI_API_KEY for the project via .env file.")
parser.add_argument('--write-dotenv', action='store_true', 
                    help='If supplied, also write a .env file in the project root directory.')
args = parser.parse_args()

# --- 2. Get API Key Input ---
# Read-Host equivalent (input is visible, like the current PowerShell script)
try:
    plain_key = input("Enter your GEMINI API key (visible): ")
except EOFError:
    print("\nError: Input cancelled.", file=sys.stderr)
    sys.exit(1)

# --- 3. Set Variable for CURRENT PROCESS (non-persistent equivalent) ---
# This makes the key available immediately in this Python session
os.environ['GEMINI_API_KEY'] = plain_key
print("GEMINI_API_KEY has been set for the current Python process.")

# --- 4. Handle .env File Creation/Update (Equivalent to WriteDotEnv) ---
if args.write_dotenv:
    
    # Determine project root (assuming script is run from a sub-directory like 'scripts')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    env_path = os.path.join(project_root, ".env")
    
    # Use the python-dotenv functions to safely handle the .env file
    try:
        from dotenv import set_key, get_key
    except ImportError:
        print("\nERROR: The 'python-dotenv' library is required to write the .env file.")
        print("Please install it: pip install python-dotenv")
        sys.exit(1)
        
    
    # Set/Update the key in the .env file
    # set_key handles creation, update, and preservation of other keys automatically
    set_key(env_path, "GEMINI_API_KEY", plain_key)
    
    print(f"\n✅ .env created/updated at {env_path}")
    
    # --- 5. Update .gitignore ---
    gitignore_path = os.path.join(project_root, ".gitignore")
    
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r+') as f:
            content = f.read()
            if ".env" not in content:
                f.write("\n.env\n")
                print("✅ .env added to .gitignore")
    else:
        with open(gitignore_path, 'w') as f:
            f.write(".env\n")
            print("✅ Created .gitignore and added .env")
            
# --- 6. Final Instructions ---
print("\nDone. To use this key in your project, ensure your code calls `from dotenv import load_dotenv; load_dotenv()` before accessing os.environ['GEMINI_API_KEY'].")