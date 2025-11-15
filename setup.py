from setuptools import setup, find_packages

setup(
    name="hangman-ai",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "matplotlib>=3.4.0",
    ],
    author="Your Name",
    description="Intelligent Hangman AI using HMM + RL",
    python_requires='>=3.7',
)