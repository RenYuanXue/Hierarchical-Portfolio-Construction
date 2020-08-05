import setuptools

with open("READ_package.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="HRP tests",
    version="0.0.1",
    author="Ren Yuan Xue, Joo Hyung Kim, Jiyoung Im",
    author_email="ryxue@uwaterloo.ca, j5im@uwaterloo.ca, jh72kim@uwaterloo.ca",
    description="An reproduction of Hierarchical Risk Parity related numerical tests",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RenYuanXue/Hierarchical-Portfolio-Construction",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)