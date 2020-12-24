import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt","r") as f:
    requirements = f.read()

setuptools.setup(
    name="opinion_aen", # Replace with your own username
    version="0.0.1",
    author="Lin Wang",
    author_email="wanglin.l.wang@mail.foxconn.com",
    description="Simplified Chinese Target Specific Opinion Retrieval",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    classifiers=[
        'Natural Language :: Chinese (Simplified)',
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=['opinion_aen'],
    package_data={'opinion_aen':['*.*','layers/*','models/*','state_dict/*']},
    python_requires='>=3.6',
    install_requires=requirements,
)
