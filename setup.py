import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dasflow",
    version="0.0.4",
    author="Hao Lv",
    author_email="lh21@apm.ac.cn",
    description="DAS event detection flow based on deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    license='MIT',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)


