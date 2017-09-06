import sys, os, datetime

import _config, _clean

from mylib import util

# Import all your steps
import a_step, b_step

##############################################################
##############################################################
util.shell_cp(_config.SRC_DIR + '_config.py', _config.RESULTS_PLACE)

start = datetime.datetime.now()
print start
##############################################################
##############################################################

a_res_dir = a_step.main(_config.DATA_DIR + '/2016-06-06/', 
  _config.OUT_PLACE, 
  run = True)

b_res_dir = b_step.main(a_res_dir, 
  _config.OUT_PLACE,
  run = True)


##############################################################
##############################################################


print '\n\nDone.\n'
end = datetime.datetime.now()
print end, '\nTotal Time:', end - start

_clean.main()