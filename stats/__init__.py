from .datasetresults import DataSetResults
from .analyses import set_op, hpp, agreement, clusterdata, agr_correlation

#import .analyses as ana

modules = [set_op, hpp, agreement, clusterdata, agr_correlation]


def parse_modules(parser):
    for module in modules:
        params = module.ARGPARSE_PARAMS
        parser.add_argument(params[0], params[1], action='store_true',
                            default=False, help=params[2])
