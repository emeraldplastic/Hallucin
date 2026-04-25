from setuptools import setup, find_packages

setup(
    name="hallucination-detector",
    version="0.1.0",
    description="Detect hallucinations in LLM responses by grounding claims against source context.",
    author="Hallucin Contributors",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "flask>=2.3.0",
    ],
    extras_require={
        "full": [
            "sentence-transformers>=2.7.0",
            "spacy>=3.7.0",
        ]
    },
    python_requires=">=3.9",
    entry_points={"console_scripts": ["hallucin=hallucination_detector.__main__:main"]},
)
