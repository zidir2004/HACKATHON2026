"""Sauvegarde du projet sur GitHub (exécuté sur le PC, avec les identifiants git de l'utilisateur).
Usage : python -u usine/src/00_git_push.py "message de commit"
"""
import subprocess, sys
REMOTE = "https://github.com/zidir2004/HACKATHON2026.git"
msg = sys.argv[1] if len(sys.argv) > 1 else "Sauvegarde"
msg += ("\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
        "Claude-Session: https://claude.ai/code/session_01MxGF6ovzwFXBmJLoWYLi4g")

def git(*a, check=True):
    r = subprocess.run(["git", *a], capture_output=True, text=True)
    print("$ git", " ".join(a), "\n", (r.stdout + r.stderr).strip(), flush=True)
    if check and r.returncode: sys.exit(r.returncode)
    return r

if git("rev-parse", "--is-inside-work-tree", check=False).returncode:
    git("init", "-b", "main")
git("config", "user.name", "Idir")
git("config", "user.email", "zidir2004@users.noreply.github.com")
if "origin" not in git("remote", check=False).stdout.split():
    git("remote", "add", "origin", REMOTE)
git("add", "-A")
git("commit", "-m", msg, check=False)
distant = git("ls-remote", "--heads", "origin", "main", check=False)
if distant.stdout.strip():
    git("pull", "origin", "main", "--allow-unrelated-histories", "--no-edit", "--no-rebase")
git("push", "-u", "origin", "main")
print("OK : poussé sur", REMOTE)
