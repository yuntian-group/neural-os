
import os
from setuptools import setup

def parse_requirements(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read().splitlines()

# Read requirements from requirements.txt
requirements = parse_requirements("requirements.txt")

setup(
    name="CSLLM",  # Replace with your package name
    version="0.1.0",  # Initial version
    # author="Your Name",
    # author_email="your.email@example.com",
    # description="Computer simulator",
    # url="https://github.com/yourusername/your-repo",
    install_requires=requirements
)