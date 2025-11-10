from setuptools import setup, find_packages

setup(
    name="retail-impact-measurement",
    version="1.0.0",
    description="Causal inference framework for measuring retail promotion impact",
    author="Dheeraj Bhaskaruni",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "statsmodels>=0.14.0",
        "scikit-learn>=1.3.0",
        "dowhy>=0.11",
        "econml>=0.15.0",
    ],
)
