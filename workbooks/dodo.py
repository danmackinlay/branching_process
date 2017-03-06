#! /usr/bin/env python

"""
See http://pydoit.org/tasks.html
"""
from doit.action import CmdAction
from doit.task import clean_targets

import sys


def gui_open_action(path):
    action = None
    if sys.platform.startswith('linux'):
        action = ["xdg-open", str(path)]
    elif sys.platform.startswith('darwin'):
        action = ["open", str(path)]
    elif sys.platform.startswith('win'):
        action = ["start", str(path)]
    return action


def task_undersampled_hawkes_latex():
    """
    see http://nbconvert.readthedocs.io/en/latest/usage.html
    """
    return dict(
        file_dep=[
            'undersampled_hawkes.ipynb',
            'undersampled_hawkes.tplx',
            'undersampled_hawkes.bib',
        ],
        targets=[
            '_paper_output/undersampled_hawkes.tex',
            '_paper_output/undersampled_hawkes.bib'
        ],
        actions=[
            'mkdir -p _paper_output',
            'ln -f undersampled_hawkes.bib _paper_output',
            'jupyter nbconvert --to latex --template=undersampled_hawkes.tplx'
            ' --FilesWriter.build_directory=_paper_output/'
            ' undersampled_hawkes.ipynb',
        ],
        clean=[
            'rm -rf _paper_output/undersampled_hawkes_files',
            clean_targets,
        ],
    )


def task_undersampled_hawkes_pdf():
    """
    """
    return dict(
        file_dep=[
            '_paper_output/undersampled_hawkes.bib',
            '_paper_output/undersampled_hawkes.tex'
            ],
        targets=[
            '_paper_output/undersampled_hawkes.pdf',
            '_paper_output/undersampled_hawkes.aux',
            '_paper_output/undersampled_hawkes.dvi',
            '_paper_output/undersampled_hawkes.bcf',
            '_paper_output/undersampled_hawkes.blg',
            '_paper_output/undersampled_hawkes.bbl',
            '_paper_output/undersampled_hawkes.run.xml',
            '_paper_output/texput.log',
            '_paper_output/q.log',
        ],
        actions=[
            CmdAction(
                'pdflatex -halt-on-error -interaction=batchmode '
                'undersampled_hawkes',
                cwd='_paper_output'),
            CmdAction(
                'bibtex undersampled_hawkes',
                cwd='_paper_output'),
            CmdAction(
                'pdflatex -halt-on-error -interaction=batchmode '
                'undersampled_hawkes',
                cwd='_paper_output'),
            CmdAction(
                'pdflatex -halt-on-error -interaction=batchmode '
                'undersampled_hawkes',
                cwd='_paper_output'),
        ],
        verbosity=1,
        clean=True,
    )


def task_undersampled_hawkes_pdf_view():
    """
    """
    return dict(
        file_dep=['_paper_output/undersampled_hawkes.pdf'],
        targets=[],
        actions=[
            gui_open_action('_paper_output/undersampled_hawkes.pdf'),
        ],
    )


def task_undersampled_hawkes_zdravko():
    """
    """
    return dict(
        file_dep=['_paper_output/undersampled_hawkes.pdf'],
        actions=[
            CmdAction(
                'rsync -av undersampled_hawkes_files '
                'undersampled_hawkes.bib undersampled_hawkes.tex '
                'undersampled_hawkes.pdf'
                ' ~/Dropbox/dan-zdravko-stuff/tex/undersampled_hawkes/',
                cwd='_paper_output'
            ),
        ],
        verbosity=2
    )
