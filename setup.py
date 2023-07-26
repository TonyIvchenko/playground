from setuptools import setup, find_packages


setup(
    name="playground",
    version="0.1.0",
    description="Playground services and utilities",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "redis>=4.5.4",
        "fastapi>=0.115.12",
        "gradio>=4.44.1,<5",
        "joblib>=1.4.2",
        "numpy>=1.26.4",
        "pandas>=2.2.3",
        "pydantic>=2.11.1",
        "scikit-learn>=1.5.2",
        "uvicorn>=0.34.0",
        "huggingface_hub>=0.35.3,<1",
    ],
    extras_require={"dev": ["pytest>=7.0"]},
    python_requires=">=3.8",
)
