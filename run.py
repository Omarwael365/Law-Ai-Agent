"""
Legal AI Assistant — Project Runner
Run this script to set up and launch the entire project.
"""

import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
VENV_STREAMLIT = PROJECT_ROOT / ".venv" / "Scripts" / "streamlit.exe"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"


def print_header():
    print()
    print("=" * 60)
    print("  Legal AI Assistant — Powered by Agentic RAG")
    print("=" * 60)
    print()


def check_venv():
    """Check if virtual environment exists."""
    if not VENV_PYTHON.exists():
        print("[!] Virtual environment not found.")
        print("    Creating one now...")
        subprocess.run([sys.executable, "-m", "venv", ".venv"], cwd=str(PROJECT_ROOT))
        print("    Installing dependencies...")
        subprocess.run([str(VENV_PYTHON), "-m", "pip", "install", "-r", "requirements.txt"], cwd=str(PROJECT_ROOT))
        print("    Done!\n")
    else:
        print("[OK] Virtual environment found.")


def check_env_file():
    """Check if .env file exists."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        print("[!] .env file not found. Copying from .env.example...")
        import shutil
        shutil.copy(PROJECT_ROOT / ".env.example", env_file)
        print("    Please edit .env and add your API keys.\n")
    else:
        print("[OK] .env file found.")


def check_vectorstore():
    """Check if vector stores have been built."""
    case_law_index = VECTORSTORE_DIR / "case_law_index"
    contracts_index = VECTORSTORE_DIR / "contracts_index"
    return case_law_index.exists() and contracts_index.exists()


def run_ingestion():
    """Run the document ingestion pipeline."""
    print("\n--- Running Document Ingestion Pipeline ---\n")
    subprocess.run([str(VENV_PYTHON), "-m", "src.ingestion.ingest"], cwd=str(PROJECT_ROOT))


def run_streamlit():
    """Launch the Streamlit web UI."""
    print("\n--- Launching Streamlit Web UI ---\n")
    print("  Open your browser at: http://localhost:8501\n")
    subprocess.run([str(VENV_STREAMLIT), "run", "app/streamlit_app.py"], cwd=str(PROJECT_ROOT))


def run_tests():
    """Run the test suite."""
    print("\n--- Running Tests ---\n")
    subprocess.run([str(VENV_PYTHON), "-m", "pytest", "tests/", "-v", "--tb=short"], cwd=str(PROJECT_ROOT))


def run_benchmark():
    """Run the evaluation benchmark."""
    print("\n--- Running Evaluation Benchmark ---\n")
    subprocess.run(
        [str(VENV_PYTHON), "-m", "src.evaluation.benchmark"],
        cwd=str(PROJECT_ROOT),
    )


def show_menu():
    """Display the main menu."""
    print("\nWhat would you like to do?\n")
    print("  1. Run everything (ingestion + launch UI)")
    print("  2. Run ingestion only")
    print("  3. Launch Streamlit UI only")
    print("  4. Run tests")
    print("  5. Run evaluation benchmark")
    print("  6. Exit")
    print()


def main():
    print_header()

    # Pre-flight checks
    check_venv()
    check_env_file()

    has_vectorstore = check_vectorstore()
    if has_vectorstore:
        print("[OK] Vector stores found (ingestion already done).")
    else:
        print("[--] Vector stores not found (ingestion needed).")

    while True:
        show_menu()
        choice = input("Enter your choice (1-6): ").strip()

        if choice == "1":
            if not has_vectorstore:
                run_ingestion()
                has_vectorstore = True
            else:
                print("\n  Vector stores already exist. Skipping ingestion.")
                print("  (Delete the vectorstore/ folder to re-ingest.)\n")
            run_streamlit()

        elif choice == "2":
            run_ingestion()
            has_vectorstore = True

        elif choice == "3":
            if not has_vectorstore:
                print("\n  Warning: Vector stores not built. Run ingestion first (option 2).")
                continue
            run_streamlit()

        elif choice == "4":
            run_tests()

        elif choice == "5":
            if not has_vectorstore:
                print("\n  Warning: Vector stores not built. Run ingestion first (option 2).")
                continue
            run_benchmark()

        elif choice == "6":
            print("\nGoodbye!")
            break

        else:
            print("\n  Invalid choice. Please enter 1-6.")


if __name__ == "__main__":
    main()
