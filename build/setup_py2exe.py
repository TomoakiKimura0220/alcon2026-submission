from setuptools import setup
import py2exe

setup(
    name="alcon2026-submission",
    version="1.0.0",

    # 自動パッケージ検出を止める
    packages=[],
    py_modules=[],

    console=[
        {
            "script": "app/run_a.py",
            "dest_base": "alcon_A",
        },
        {
            "script": "app/run_b.py",
            "dest_base": "alcon_B",
        },
        {
            "script": "app/run_c.py",
            "dest_base": "alcon_C",
        },
    ],

    options={
        "py2exe": {
            "packages": [
                "torch",
                "torchvision",
                "PIL",
                "numpy",
            ],
            "excludes": [
                "tkinter",
                "matplotlib",
                "pandas",
                "pytest",
                "IPython",
            ],
            "compressed": True,
            "optimize": 1,
            "bundle_files": 3,
        }
    },

    zipfile="library.zip",
)