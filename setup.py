import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="n2v_dataset_iterator",
    version="0.0.1",
    author="Jean Ollion",
    author_email="jean.ollion@polytechnique.org",
    description="noise2void utils for compatibility with dataset_iterator and tf.keras",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jeanollion/n2v_dataset_iterator.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
    install_requires=['numpy', 'tensorflow']
)
