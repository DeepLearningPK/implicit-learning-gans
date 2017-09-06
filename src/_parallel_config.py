import _config

# Which script are we parallelizing?
P_SCRIPT = 'scriptname'

SPLITS = 8

# How to split input files?
#   line : Splits consist of subsets of lines 
#           grabbed from all input files
#   file : Splits consist of subsets of files
SPLIT_TYPE = 'line'     # 'line' or 'file'

# regex to match input filenames
REGEX_FILTER = '**'


# When splitting by line, ensures that each file's number of lines
# is divisible by this number.
# Ensures that a multi-line "objects" aren't split across files.
# Ex: 4 for fastq, 2 for fasta, 1 if unneeded
LINES_DIVISOR = 2