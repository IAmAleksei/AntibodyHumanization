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
    package_data={'humanization': ['humanization/config.yaml']},
    include_package_data=True,
    install_requires=[
        'pandas',
        'configloader',
        'catboost',
        'blosum'
    ],
    license='http://www.apache.org/licenses/LICENSE-2.0',
)
