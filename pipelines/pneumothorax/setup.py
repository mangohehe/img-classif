from setuptools import setup, find_packages

setup(
    name="pipelines",
    version="0.1.0",
    description="Pneumothorax model reference service on a Ray-based cluster on GCP",
    author="Feng Gao",
    url="https://github.com/mangohehe/img-classif",  # Update if applicable
    packages=find_packages(include=["src", "src.*", "models", "models.*"]),
    install_requires=[
        "ray[serve]",
        "torch",
        "numpy",
        "requests",
        "tqdm",
        "starlette",
        "gcsfs",
        "opencv-python",
        "albumentations",
        "PyYAML",
        "cryptography",
        "google-cloud-storage",
        "google-api-python-client==1.7.8"
    ],
    entry_points={
        "console_scripts": [
            "pneumothorax-pipeline=src.main:main",
        ],
    },
    python_requires=">=3.7",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
