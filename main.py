"""
Neural FOXP2 — Multi-Language Launcher

Dispatches to the Hindi or Spanish pipeline based on --language.
All other CLI arguments are forwarded transparently.
"""
import subprocess
import sys
import os


USAGE = """
Neural FOXP2: Language Steering via SAE Interventions

Usage:
  python main.py --language {hindi,spanish} [PIPELINE_OPTIONS...]

Required:
  --language {hindi,spanish}   Target language pipeline to run

All other arguments are forwarded to the selected pipeline.
Run  python main.py --language hindi --help  to see pipeline options.

Examples:
  python main.py --language hindi   --hf-token YOUR_TOKEN
  python main.py --language spanish --hf-token YOUR_TOKEN
  python main.py --language hindi   --stage 1 --layers 18 --hf-token YOUR_TOKEN
  python main.py --language spanish --resume-from ./outputs/stage1_checkpoint.pkl
""".strip()


def main():
    argv = sys.argv[1:]

    # Show our own help only when no --language is present
    if not argv or ("--help" in argv and "--language" not in argv):
        print(USAGE)
        sys.exit(0)

    # Extract --language value manually so all other flags pass through
    if "--language" not in argv:
        print("Error: --language is required.  Use --help for usage.\n")
        print(USAGE)
        sys.exit(1)

    idx = argv.index("--language")
    if idx + 1 >= len(argv) or argv[idx + 1] not in ("hindi", "spanish"):
        print("Error: --language must be 'hindi' or 'spanish'.\n")
        print(USAGE)
        sys.exit(1)

    language = argv[idx + 1]
    remaining = argv[:idx] + argv[idx + 2:]  # everything except --language & its value

    # Resolve paths
    root_dir = os.path.dirname(os.path.abspath(__file__))
    language_dir = os.path.join(root_dir, language)
    language_main = os.path.join(language_dir, "main.py")

    if not os.path.isfile(language_main):
        print(f"Error: {language_main} not found.")
        sys.exit(1)

    # Build command and environment
    cmd = [sys.executable, language_main] + remaining

    # Add root dir to PYTHONPATH so language folders can import shared config.py
    env = os.environ.copy()
    python_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root_dir + (os.pathsep + python_path if python_path else "")

    print("=" * 70)
    print(f"Neural FOXP2 — Launching {language.upper()} pipeline")
    print("=" * 70)
    print(f"Working directory : {language_dir}")
    print(f"Command           : {' '.join(cmd)}")
    print("=" * 70 + "\n")

    # Run the language-specific pipeline from within its directory
    result = subprocess.run(cmd, cwd=language_dir, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
