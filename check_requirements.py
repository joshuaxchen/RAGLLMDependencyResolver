import argparse
import os
import shutil
import subprocess
import sys
import tempfile

def create_virtualenv(venv_dir):
    """Create a virtual environment in the given directory."""
    subprocess.check_call([sys.executable, "-m", "venv", venv_dir])

def get_pip_executable(venv_dir):
    """Return the path to the pip executable inside the virtual environment."""
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "pip.exe")
    return os.path.join(venv_dir, "bin", "pip")

def install_and_check(req_files):
    """Install multiple requirements files in one venv and run pip check."""
    venv_dir = tempfile.mkdtemp(prefix="req_check_")
    try:
        print(f"\nChecking {', '.join(req_files)} in virtualenv at {venv_dir}...")
        create_virtualenv(venv_dir)
        pip_exe = get_pip_executable(venv_dir)

        # Build pip install command with all -r flags
        install_cmd = [pip_exe, "install"]
        for rf in req_files:
            install_cmd.extend(["-r", rf])
        install_proc = subprocess.run(
            install_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        if install_proc.returncode != 0:
            return f"ERROR during install:\n{install_proc.stdout.strip()}"

        # Run pip check for dependency conflicts
        check_proc = subprocess.run(
            [pip_exe, "check"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        if check_proc.returncode == 0:
            return "No dependency issues found."
        return f"Issues detected:\n{check_proc.stdout.strip()}"

    finally:
        shutil.rmtree(venv_dir, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(
        description="Check one or multiple requirements.txt for dependency issues"
    )
    parser.add_argument(
        "requirements",
        nargs="+",
        help="Paths to requirements.txt files"
    )
    parser.add_argument(
        "--combined", "-c",
        action="store_true",
        help="Install all requirement files together in one environment"
    )
    args = parser.parse_args()

    # Validate file existence
    missing = [f for f in args.requirements if not os.path.isfile(f)]
    if missing:
        for f in missing:
            print(f"File not found: {f}")
        sys.exit(1)

    if args.combined:
        # Combined install and check
        issues = install_and_check(args.requirements)
        print(f"Results for combined install of {len(args.requirements)} files:\n{issues}")
    else:
        # Individual checks
        for req_file in args.requirements:
            issues = install_and_check([req_file])
            print(f"Results for {req_file}:\n{issues}")

if __name__ == "__main__":
    main()
