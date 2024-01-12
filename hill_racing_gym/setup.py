from setuptools import setup,find_packages

setup(
    name="hill_racing_env",
    version="0.0.1",
    install_requires=["gymnasium", "pygame"],
    packages=find_packages(),
)