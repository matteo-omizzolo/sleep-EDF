from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sleep-edf-hdp-hmm",
    version="0.1.0",
    author="Matteo Omizzolo",
    author_email="",
    description="Hierarchical Dirichlet Process HMM for unsupervised sleep staging",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/matteo-omizzolo/sleep-EDF",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.6b0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sleep-edf-download=scripts.download_data:main",
            "sleep-edf-preprocess=scripts.preprocess_all:main",
            "sleep-edf-train=scripts.run_hdp_hmm_sticky:main",
        ],
    },
)
