"""Question-Space Driven Knowledge Compilation pipeline.

A prototype system that treats documents as containers of answer-bearing
fragments, discovers the questions those fragments can answer, organizes those
questions into a latent question space (Qspace), discovers fragment classes as
question subspaces, and compiles retrieval-optimized trees that minimize
expected answer path length.
"""

__version__ = "0.1.0"
