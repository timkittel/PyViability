# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS      =
SPHINXBUILD     = sphinx-build
SPHINXBUILD_API = sphinx-apidoc
SPHINXPROJ      = pyviability
SOURCEDIR       = source
BUILDDIR        = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).

api:
	@$(SPHINXBUILD_API) --force --separate --implicit-namespaces --output-dir="source/" "../pyviability/" "../pyviability/periodic_kdtree.py"

html: clean api
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

