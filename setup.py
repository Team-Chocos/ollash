from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
setup(
    name='ollash',
    version='0.1.5',
    author='Sparsh Chakraborty',
    author_email='sparsh.chakraborty07@gmail.com',
    description='Convert natural language into safe Terminal commands using Ollama.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/codexx07/ollash',
    packages=find_packages(),
    python_requires='>=3.7',
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'ollash=ollash.__main__:main'
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Topic :: Utilities',
    ],
)
