# setup.py
from setuptools import setup, find_packages

setup(
    name="mutator_evo",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "backtrader",
        "numpy",
        "pandas",
        "matplotlib",
        "plotly",
        "structlog",
        "sentry_sdk",
        "dill"
    ],
    entry_points={
        "console_scripts": [
            "mutator-evo=mutator_evo.scripts.run_evolution:main"
        ]
    },
)