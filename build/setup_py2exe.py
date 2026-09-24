from setuptools import setup
import py2exe

setup(
    console=[
        {"script": "app/run_a.py", "dest_base": "alcon_A"},
        {"script": "app/run_b.py", "dest_base": "alcon_B"},
        {"script": "app/run_c.py", "dest_base": "alcon_C"},
    ],
    options={"py2exe": {
        "packages": ["torch", "torchvision", "PIL", "numpy"],
        "excludes": ["tkinter", "matplotlib", "pandas", "pytest", "IPython"],
        "optimize": 1,
        "compressed": True,
        "bundle_files": 3,
    }},
    zipfile="library.zip",
)
