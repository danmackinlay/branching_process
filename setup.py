try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

long_description = """
branching_process
"""

config = dict(
    description='hawkes process fitting',
    author='Dan MacKinlay',
    url='URL to get it at.',
    download_url='Where to download it.',
    author_email='My email.',
    version='0.1',
    install_requires=[
        'nose',
        'scipy',
        'numpy',
        'seaborn',
        'pandas',
    ],
    packages=['branching_process'],
    scripts=[],
    name='branching_process',
    # # see https://python-packaging.readthedocs.io/en/latest/non-code-files.html
    # package_data=dict(
    #     branching_process= ['datasets'],
    # ),
    # include_package_data=True
)

setup(**config)
