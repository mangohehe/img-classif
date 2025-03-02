from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="image_classif",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,  # Install all dependencies
    entry_points={
        "console_scripts": [
            "ray_infer=my_ray_project.main:main",
        ],
    },
)