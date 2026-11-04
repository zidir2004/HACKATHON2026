"""Vérifie l'environnement Python du PC (versions, CPU, RAM, GPU)."""
import sys, os, platform, importlib, subprocess
print("python", sys.version); print("os", platform.platform()); print("cpu", os.cpu_count())
try:
    import ctypes
    class M(ctypes.Structure):
        _fields_=[("l",ctypes.c_ulong),("t",ctypes.c_ulong),("tp",ctypes.c_ulonglong),("ap",ctypes.c_ulonglong)]+[(f"x{i}",ctypes.c_ulonglong) for i in range(5)]
    m=M(); m.l=ctypes.sizeof(M); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    print("ram_go", round(m.tp/2**30,1), "libre_go", round(m.ap/2**30,1))
except Exception as e: print("ram ?", e)
try: print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv"],capture_output=True,text=True).stdout)
except Exception as e: print("gpu: nvidia-smi absent", e)
for lib in ["pandas","numpy","sklearn","xgboost","lightgbm","catboost","shap","matplotlib","pyarrow"]:
    try: print(lib, importlib.import_module(lib).__version__)
    except Exception as e: print(lib, "MANQUANT", e)
