from setuptools import find_packages, setup

setup(
    name="houtrellis",
    version="1.0.0",
    description="HouTrellis: Multiplatform 3D Generative AI Pipeline Suite for Houdini",
    author="Hristo Velev & Zed Agent",
    packages=find_packages(),
    install_requires=["fastapi", "uvicorn", "pydantic", "requests", "pyyaml"],
    entry_points={
        "console_scripts": [
            "houtrellis=backend.cli:main",
        ],
    },
    python_requires=">=3.10",
)
