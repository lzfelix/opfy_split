# coding:utf-8
import os
import subprocess
import logging


def invoke(opf_path, opf_command, *args):
    """Wraps subprocess to call OPF programs.

    :param opf_path: Path to the OPF bin/ folder.
    :param opf_command: Which OPF program to run.
    :param *args: The parameters for opf_command.
    :returns None
    :raises RuntimeError if there's an error while running opf_command.
    """

    # command is opf_path/opf_command followed by a list of string parameters
    c = [os.path.join(opf_path, opf_command)] + [str (arg) for arg in args]

    try:
        result = subprocess.run(c, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except:
        raise RuntimeError('Failed to invoke OPF command {}. Is the provided '
                           'OPF_PATH correct? ({})'.format(opf_command, opf_path))

    if result.returncode != 0:
        logging.error(result.stderr.decode('utf-8'))
        raise RuntimeError('Failed to run OPF command {}'.format(opf_command))


def create_dir(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)


def get_opf_path(opf_path):
    return os.path.expanduser(opf_path) if opf_path else os.environ['OPF_PATH']
