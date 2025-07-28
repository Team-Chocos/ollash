from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
setup(
    name='ollash',
    version='1.0.0',
    author='Team-Chocos',
    author_email='teamcornflakesxx@gmail.com',
    description='Convert natural language into safe Terminal commands using Ollama.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/team-chocos/ollash',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "ollash": ["config.yaml"]
    },
    entry_points={
        'console_scripts': [
            'ollash=ollash.__main__:main',
        ],
    },

    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Topic :: Utilities',
    ],
)
