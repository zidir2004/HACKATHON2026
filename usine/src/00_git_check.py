"""Vérifie que git est installé sur le PC et affiche sa configuration (sans secrets)."""
import subprocess
for cmd in (["git", "--version"], ["git", "config", "--global", "user.name"], ["git", "config", "--global", "user.email"], ["git", "config", "--global", "credential.helper"]):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(" ".join(cmd), "->", (r.stdout or r.stderr).strip())
    except FileNotFoundError:
        print(" ".join(cmd), "-> git introuvable")
