from setuptools import setup

setup(
    name='emotion-extraction',
    version='0.1.0',
    packages=['emotion_extraction'],
    entry_points={
        'console_scripts': [
            'emotion-extraction = emotion_extraction.__main__:main'
        ]
    })
