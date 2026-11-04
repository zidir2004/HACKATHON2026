"""Lanceur de tâches de l'usine ML (à exécuter sur le PC Windows).

Usage (depuis le dossier Hackathon) :  python usine\\runner.py

Il surveille usine/queue/ : chaque fichier <nom>.job contient une ligne d'arguments
pour python (ex. « -u usine/src/03_train.py --exp E01 »).
- Sortie dans usine/logs/<nom>.log, code retour dans usine/logs/<nom>.status.
- Un job dont la ligne commence par « #gpu » utilise la carte graphique : un seul job
  GPU tourne à la fois (les autres attendent), pour éviter les blocages.
- Pour annuler un job en cours : créer usine/queue/<nom>.kill (code retour -9).
Ctrl+C pour arrêter le lanceur (et ses jobs).
"""
import pathlib, shlex, subprocess, sys, time, datetime, os

ROOT = pathlib.Path(__file__).resolve().parents[1]
QUEUE, LOGS = ROOT / "usine" / "queue", ROOT / "usine" / "logs"
QUEUE.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
running = {}  # nom -> (process, fichier log, gpu ?)

def now(): return f"{datetime.datetime.now():%H:%M:%S}"

# Jobs restés « .running » d'une session précédente : marqués orphelins
for r in QUEUE.glob("*.running"):
    r.rename(r.with_suffix(".orphan")); (LOGS / f"{r.stem}.status").write_text("orphan")
print(f"Runner v2 démarré ({sys.executable}) dans {ROOT}. Ctrl+C pour arrêter.", flush=True)

try:
    while True:
        gpu_occupe = any(g for _, _, g in running.values())
        for job in sorted(QUEUE.glob("*.job")):
            ligne = job.read_text(encoding="utf-8").strip()
            gpu = ligne.startswith("#gpu")
            if gpu:
                if gpu_occupe:
                    continue  # attend que la carte graphique soit libre
                ligne = ligne[len("#gpu"):].strip()
            name = job.stem
            job.rename(job.with_suffix(".running"))
            log = open(LOGS / f"{name}.log", "w", encoding="utf-8")
            p = subprocess.Popen([sys.executable, *shlex.split(ligne)], cwd=ROOT,
                                 stdout=log, stderr=subprocess.STDOUT, env=env)
            running[name] = (p, log, gpu); gpu_occupe = gpu_occupe or gpu
            print(f"{now()} lancé  {name}{' [GPU]' if gpu else ''}: {ligne}", flush=True)
        for kill in QUEUE.glob("*.kill"):
            if kill.stem in running:
                running[kill.stem][0].kill(); print(f"{now()} annulé {kill.stem}", flush=True)
            kill.unlink()
        for name, (p, log, gpu) in list(running.items()):
            if p.poll() is not None:
                log.close()
                (LOGS / f"{name}.status").write_text(str(p.returncode))
                r = QUEUE / f"{name}.running"
                if r.exists(): r.rename(QUEUE / f"{name}.done")
                print(f"{now()} fini   {name} (code {p.returncode})", flush=True)
                del running[name]
        (LOGS / "_heartbeat").write_text(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} v2 running={list(running)}")
        time.sleep(2)
except KeyboardInterrupt:
    for name, (p, log, gpu) in running.items():
        p.kill(); print(f"arrêt : {name} tué", flush=True)
