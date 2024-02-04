from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="pafik",
    version="0.1.0",
    author="chiahanlu",
    author_email="ducklyu0301@gmail.com",
    description="An data-driven posture-aware IK solver",
    keywords="data-driven, inverse kinematics, posture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    url="https://github.com/duckhanson/PAFIK",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9.12",
        "Operating System :: OS Independent",
    ],
    install_requires=required,
    python_requires=">=3.9.12",
)
