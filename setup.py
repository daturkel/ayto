from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="ayto",
    version="0.1.0",
    description="A tool for calculating couple probabilities for the TV series Are You The One?",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/daturkel/ayto",
    author="Dan Turkel",
    author_email="daturkel@gmail.com",
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        # Pick your license as you wish
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="probability, statistics, are-you-the-one",
    packages=find_packages(),
    python_requires=">=3.8, <4",
    install_requires=["pandas"],  # Optional
    # If there are data files included in your packages that need to be
    # installed, specify them here.
    # package_data={  # Optional
    #     "sample": ["package_data.dat"],
    # },
    project_urls={  # Optional
        "Bug Reports": "https://github.com/daturkel/ayto/issues",
        "Author's website": "https://danturkel.com",
        "Source": "https://github.com/daturkel/ayto/",
    },
)
