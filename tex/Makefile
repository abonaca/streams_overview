LATEX       = pdflatex -interaction=nonstopmode -halt-on-error
RM          = rm -rf
TMP_SUFFS   = .pdf .aux .bbl .blg .log .dvi .ps .eps .out .brf Notes.bib .synctex.gz .spl
NAME = streams-overview

all: ${NAME}.pdf

${NAME}.pdf: ${NAME}.tex preamble.tex refs.bib
	${LATEX} ${NAME}
	bibtex ${NAME}
	${LATEX} ${NAME}
	${LATEX} ${NAME}


clean:
	${RM} $(foreach suff, ${TMP_SUFFS}, ${NAME}${suff})
