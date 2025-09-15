import subprocess, sys
cmd = [sys.executable, "-m", "src.main", "--epochs", "1"]
raise SystemExit(subprocess.call(cmd))