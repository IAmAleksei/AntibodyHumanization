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
    data_files=[('humanization', ['humanization/config.yaml'])],
    install_requires=[
        'pandas',
        'configloader',
        'catboost',
        'blosum',
        'tqdm'
    ],
    license='http://www.apache.org/licenses/LICENSE-2.0',
)
