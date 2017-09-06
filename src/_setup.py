from mylib import util
import _config

# Write out the code dependency
util.code_dependency(_config.SRC_DIR)

# Write source README
with open(_config.SRC_DIR + 'README', 'w') as f:
  f.write('''README
Auxiliary scripts begin with _, main scripts do not.

_config.py lists configurable parameters for the collection of scripts in this src directory. Within this, there are four key directories specified:

  /data/ is expected to store raw data only.
  /out/ holds processed data. It is not designed to be long-lasting storage. Scripts will freely overwrite and delete data in /out/.
  /results/ holds final analysis results files that are meant to be kept.
  /src/ is this directory and stores code.

_clean.py will delete all files generated in /out/.

_dependencies.txt is created by running this script (_setup.py) and describes the input directory dependencies for all main scripts. Typically, the scripts are named such that b_analysis will use the output directory of a_analysis, and so on. Default dependencies are explicitly described in each script in the DEFAULT_INP_DIR variable and is used when running a script in isolation.

_runall.py will run the analysis pipeline, overwriting any files in /out/ and /results/. If you wish to store results, it is recommended to copy the results directory content to another folder before re-running _runall.py.

  Each time _runall.py is run, it will copy the current _config.py file into the results folder for reference.

  _runall.py also passes input/output directories to each script in the pipeline dynamically, instead of relying on each script's default input directory. This means that a single script's output directory can be modified and all child scripts that depend on that directory will automatically find that modified directory in _runall.py, but not when the child scripts are run in isolation.

All of a script's output is contained within the folder /out/script_name/ and /results/script_name/.

Scripts are written with a flexible main() function that accepts input/output paths as parameters, enabling the function to be called with custom input/output directories via other scripts or the interactive python terminal.

    ''')