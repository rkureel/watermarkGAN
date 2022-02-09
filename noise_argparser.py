import argparse
import argparse

from numpy import require
from noise_layers.identity import Identity


class NoiseArgParser(argparse.Action):
    def __ini__(
        self,
        option_strings,
        dest,
        nargs=None,
        const=None,
        default=None,
        type=None,
        choices=None,
        required=False,
        help=None,
        metavar=None
        ):
        argparse.Action.__init__(
            self,
            option_strings=option_strings,
            dest=dest,
            nargs=nargs,
            const=const,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar
        )
    
    def __call__(self, parser, namespace, values, option_string=None):
        layers = []
        split_commands = values[0].split("+")
        for command in split_commands:
            command = command.replace(' ', '')
            if command[:len("identity")] == "identity":
                pass
            else:
                raise ValueError(f"Command not recognized: \n{command}")
        setattr(namespace, self.dest, layers)