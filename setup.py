from setuptools import setup, find_packages

setup(name='PAD-TRO',
    author="Jushan Chen",
    author_email="chenj72@rpi.edu",
    packages=find_packages(include="PAD-TRO"),
    version='0.0.1',
    install_requires=[
        'gym', 
        'pandas', 
        'seaborn', 
        'matplotlib', 
        'imageio',
        'control', 
        'tqdm', 
        'tyro', 
        'meshcat', 
        'sympy', 
        'gymnax',
        'jax', 
        'distrax', 
        'gputil', 
        'jaxopt'
        ]
)
