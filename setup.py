from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="healthguard-ai",
    version="1.0.0",
    description="HealthGuard AI: a Flask machine learning app for heart disease risk assessment and patient record tracking.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="B.Sujay",
    author_email="sujayboddu@gmail.com",
    url="https://github.com/Sujay6612/HEALTHGUARD-AI",
    license="MIT",
    packages=find_packages(),
    py_modules=["app", "train_model"],
    include_package_data=True,
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "Flask>=2.0.0",
        "Werkzeug>=2.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
