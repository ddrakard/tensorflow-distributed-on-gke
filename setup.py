from setuptools import setup

with open("README.md", 'r') as file:
    long_description = file.read()

setup(
    name='Transformer distributed training',
    version='0.1',
    description=(
        'A demonstration of distributed training of a Transformer model using '
        + 'GKE on GCP.'
    ),
    long_description=long_description,
    url='',
    author='David Drakard',
    author_email='research@ddrakard.com',
    install_requires=[
        'tensorflow',
        'tensorflow_datasets',
        'tensorflow-text',
        'kubernetes',
        'pyyaml',
        'google-cloud-storage'
    ],
    zip_safe=False
)
