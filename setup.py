from setuptools import setup, find_packages


setup(
    name="playground",
    version="0.1.0",
    description="Playground services and utilities",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["redis>=4.5.4"],
    extras_require={"dev": ["pytest>=7.0"]},
    python_requires=">=3.8",
)
