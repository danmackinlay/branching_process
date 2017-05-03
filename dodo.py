#! /usr/bin/env python

"""
See http://pydoit.org/tasks.html
"""
from doit.action import CmdAction
from doit.task import clean_targets

import sys


def gui_open_action(pth):
    action = None
    if sys.platform.startswith('linux'):
        action = ["xdg-open", str(pth)]
    elif sys.platform.startswith('darwin'):
        action = ["open", str(pth)]
    elif sys.platform.startswith('win'):
        action = ["start", str(pth)]
    return action


def _task_html(pth):
    """
    see http://nbconvert.readthedocs.io/en/latest/usage.html
    """
    return dict(
        file_dep=[
            'docs/{pth}.ipynb'.format(pth=pth),
            'docs/html.tpl'.format(pth=pth),
        ],
        #    + [ str(p) for p in pathlib.Path('docs/ext_media').glob('*')],
        targets=[
            'docs/{pth}.html'.format(pth=pth),
            # 'docs/refs.bib'.format(pth=pth)
        ],
        actions=[
            # 'mkdir -p docs',
            # 'ln -f docs/refs.bib docs/'.format(pth=pth),
            'jupyter nbconvert --to html '
            # '--template=docs/{pth}_html.tpl '
            '--FilesWriter.build_directory=docs/ '
            'docs/{pth}.ipynb'.format(pth=pth),
        ],
        clean=[
            # 'rm -rf docs/{pth}_files',
            clean_targets,
        ],
    )


def _task_latex(pth):
    """
    see http://nbconvert.readthedocs.io/en/latest/usage.html
    """
    return dict(
        file_dep=[
            'docs/{pth}.ipynb'.format(pth=pth),
            'docs/{pth}_print.tplx'.format(pth=pth),
            'docs/refs.bib'.format(pth=pth),
            # 'docs/ext_media/',
        ],
        targets=[
            '_paper_output/{pth}.tex'.format(pth=pth),
            '_paper_output/refs.bib'.format(pth=pth)
        ],
        actions=[
            'mkdir -p _paper_output',
            'rm -rf _paper_output/{pth}_files',
            'ln -f docs/refs.bib _paper_output'.format(pth=pth),
            'jupyter nbconvert --to latex --template=docs/{pth}_print.tplx '
            '--FilesWriter.build_directory=_paper_output/ '
            'docs/{pth}.ipynb'.format(pth=pth),
        ],
        clean=[
            'rm -rf _paper_output/{pth}_files',
            clean_targets,
        ],
    )


def _task_pdf(pth):
    """
    """
    return dict(
        file_dep=[
            '_paper_output/refs.bib'.format(pth=pth),
            '_paper_output/{pth}.tex'.format(pth=pth)
            ],
        targets=[
            '_paper_output/{pth}.pdf'.format(pth=pth),
            '_paper_output/{pth}.aux'.format(pth=pth),
            '_paper_output/{pth}.dvi'.format(pth=pth),
            '_paper_output/{pth}.bcf'.format(pth=pth),
            '_paper_output/{pth}.blg'.format(pth=pth),
            '_paper_output/{pth}.bbl'.format(pth=pth),
            '_paper_output/{pth}.run.xml'.format(pth=pth),
            '_paper_output/texput.log',
            '_paper_output/q.log',
        ],
        actions=[
            CmdAction(
                'pdflatex -halt-on-error -interaction=batchmode '
                '{pth}'.format(pth=pth),
                cwd='_paper_output'),
            CmdAction(
                'bibtex '
                '{pth}'.format(pth=pth),
                cwd='_paper_output'),
            CmdAction(
                'pdflatex -halt-on-error -interaction=batchmode '
                '{pth}'.format(pth=pth),
                cwd='_paper_output'),
            CmdAction(
                'pdflatex -halt-on-error -interaction=batchmode '
                '{pth}'.format(pth=pth),
                cwd='_paper_output'),
        ],
        verbosity=1,
        clean=True,
    )


def _task_view_pdf(pth):
    """
    """
    return dict(
        file_dep=['_paper_output/{pth}.pdf'.format(pth=pth)],
        targets=[],
        actions=[
            gui_open_action('_paper_output/{pth}.pdf'.format(pth=pth)),
        ],
    )


def _task_zdravko(srcpth, destpth):
    """
    """
    return dict(
        file_dep=[
            '_paper_output/{srcpth}.pdf'.format(srcpth=srcpth),
            '_paper_output/{srcpth}.tex'.format(srcpth=srcpth),
            '_paper_output/refs.bib'
        ],
        actions=[
            'mkdir -p ~/Dropbox/dan-zdravko-stuff/tex/{destpth}/'.format(
                destpth=destpth
            ),
            CmdAction(
                'rsync -av {srcpth}_files '
                'refs.bib {srcpth}.tex '
                '{srcpth}.pdf'
                ' ~/Dropbox/dan-zdravko-stuff/tex/{destpth}/'.format(
                    srcpth=srcpth,
                    destpth=destpth
                ),
                cwd='_paper_output'
            ),
        ],
        verbosity=2
    )


def task_latex_chapter_sparse_hawkes():
    return _task_latex('chapter_sparse_hawkes')


def task_pdf_chapter_sparse_hawkes():
    return _task_pdf('chapter_sparse_hawkes')


def task_view_pdf_chapter_sparse_hawkes():
    return _task_view_pdf('chapter_sparse_hawkes')


def task_zdravko_chapter_sparse_hawkes():
    return _task_zdravko('chapter_sparse_hawkes', 'sparse_hawkes')


def task_html_chapter_sparse_hawkes():
    return _task_html('chapter_sparse_hawkes')


def task_html_intro_to_cts_hawkes():
    return _task_html('intro_to_cts_hawkes')
