from setuptools import setup

setup(
    name='rm_flatness_nav', version='0.1.0',
    packages=['rm_flatness_nav'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/rm_flatness_nav']),
        ('share/rm_flatness_nav', ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy'], zip_safe=True,
    maintainer='Mohammad Ahmed Khan', maintainer_email='student@example.com',
    description='RoboMaster floor-tag flatness pursuit assessment implementation',
    license='Apache-2.0',
    entry_points={'console_scripts': [
        'driver = rm_flatness_nav.driver:main',
        'planner = rm_flatness_nav.planner:main',
        'logger = rm_flatness_nav.logger:main',
        'plot_log = rm_flatness_nav.plot_log:main',
    ]},
)
