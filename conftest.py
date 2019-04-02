'''File for configuring py.test tests'''

import pytest
try:
    import data_IO
    NO_VTK = False
except ModuleNotFoundError:
    NO_VTK = True

def pytest_addoption(parser):
    '''Adds parser options'''
    parser.addoption('--runslow', action='store_true', default=False, 
                     help='run slow tests')
    
def pytest_collection_modifyitems(config, items):
    '''If test is marked with the pytest.mark.slow decorator, mark it to be
    skipped, unless the --runslow option has been passed.'''
    if not config.getoption("--runslow"):
        # --runslow not given in cli: skip slow tests
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    # skip vtk tests if unable to import vtk
    if NO_VTK:
        skip_vtk = pytest.mark.skip(reason="could not load VTK")
        for item in items:
            if "vtk" in item.keywords:
                item.add_marker(skip_vtk)
    