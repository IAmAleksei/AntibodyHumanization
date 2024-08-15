import setuptools

setuptools.setup(
    name='antibody-humanizer',
    version="1.0.0",
    author='Alexey Shishkin',
    author_email='alexxxshishkin@yandex.ru',
    description='Antibody humanizer',
    url='https://github.com/Alexvsalexvsalex/AntibodyHumanization',
    packages=[
        'humanization'
    ],
    data_files=[('humanization', ['humanization/common/config.yaml'])],
    install_requires=[
        'pandas==2.1.1',
        'configloader',
        'catboost==1.2.2',
        'blosum',
        'tqdm',
        'scikit-learn==1.3.1',
        'PyYAML',
        'numpy==1.26.3',
        'transformers==4.32.1',
        'rjieba==0.1.11',
        'torch==2.2.1',
        'biopython==1.78',
        'tabulate',
    ],
    license='http://www.apache.org/licenses/LICENSE-2.0',
)
