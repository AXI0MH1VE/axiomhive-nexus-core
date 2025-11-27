from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="axiomhive-nexus-core",
    version="1.1.0",
    author="Alexis M. Adams",
    author_email="devdollzai@gmail.com",
    description="Deterministic AI with cryptographic proof of correctness",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AXI0MH1VE/axiomhive-nexus-core",
    project_urls={
        "Website": "https://axiomhive.org",
        "Documentation": "https://axiomhive.org/post/expert-architectural-review-axiom-hive-and-the-foundation-of-deterministic-ai",
        "Medium": "https://medium.com/@devdollzai",
        "Twitter": "https://twitter.com/devdollzai",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "isort>=5.12.0",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "axiomhive=axiomhive_nexus:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)