from setuptools import setup

setup(
    name="dlt",
    version="0.0.1",
    packages=["dlt"],
    python_requires=">=3.11",
    install_requires=[
        "dataclasses-json",
        "jax[cpu];platform_system=='Darwin'",
        "jax[cuda12];platform_system=='Linux'",
        "transformers",
        "datasets",
        "evaluate",
        "flax",
        "nltk",
        "matplotlib",
        "tensorflow",
        "rouge_score",
    ],
)
