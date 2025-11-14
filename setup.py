"""
Setup script for Crop Analysis AI System
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="crop-analysis-ai",
    version="2.1.0",
    author="Agricultural AI Research Team",
    author_email="research@cropanalysis.ai",
    description="AI-powered satellite image analysis for crop yield prediction and land use classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Niranjan117/AI-Model",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "gpu": [
            "tensorflow-gpu>=2.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "crop-analysis=run_server:main",
            "train-model=model_trainer:main",
            "test-apis=test_real_apis:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.txt"],
    },
    keywords="agriculture, ai, machine learning, satellite imagery, crop analysis, yield prediction",
    project_urls={
        "Bug Reports": "https://github.com/Niranjan117/AI-Model/issues",
        "Source": "https://github.com/Niranjan117/AI-Model",
        "Documentation": "https://github.com/Niranjan117/AI-Model/wiki",
    },
)