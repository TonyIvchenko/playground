from setuptools import setup, find_packages


setup(
    name="private",
    version="1.0.0",
    package_dir = {"": "src"},
    packages=find_packages(),
    install_requires=[
        ""
    ],
    setup_requires=['pytest-runner', 'flake8'],
    tests_require=['pytest'],
)