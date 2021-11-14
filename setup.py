import os
from setuptools import setup
from setuptools import find_packages

ld = {}
if os.path.exists("README.md"):
    ld['filename'] = "README.md"
    ld['content_type'] = "text/markdown"
elif os.path.exists("readme_src.org"):
    ld['filename'] = "readme_src.org"
    ld['content_type'] = "text/plain"

with open(file=ld['filename'], mode="r") as readme_f:
    ld['data'] = readme_f.read()

setup(
    # Metadata
    name="ollam",
    author="Philipp Denzel",
    author_email="phdenzel@gmail.com",
    version="0.0.dev2",
    description=("An LSTM NLP neural net which can easily be trained to write poems!"),
    long_description=ld['data'],
    long_description_content_type=ld['content_type'],
    license='GNU General Public License v3.0',
    url="https://github.com/phdenzel/ollam",
    keywords="machine learning, natural language processing, text generation, poems",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],

    # Package
    install_requires=['numpy',
                      'scipy',
                      'matplotlib',
                      'sklearn',
                      'tensorflow',
                      'tensorflow-gpu'],
    package_dir={"": "ollam"},
    packages=find_packages(where='ollam'),
    py_modules=['ollam'],
    python_requires=">=3.6",
    entry_points={
        'console_scripts': [
            'ollam = ollam.__main__:main',
        ],
    },

)
