from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='pafik',
    version='0.0.10',
    description="An data-driven posture-aware IK solver",
    package_dir={"": "pafik"},
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/duckhanson/PAFIK",
    author="chiahanlu",
    author_email="ducklyu0301@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9.12",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "warp-lang", 'usd-core',
                      'urdfpy', 'pytorch_lightning', 'torchdiffeq'],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.9.12",
)
