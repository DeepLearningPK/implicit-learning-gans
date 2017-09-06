# Cleans temporary files from /out/ dir
# Final output should be in /results/ dir

import _config, os
from subprocess import call

def clean():
  print 'Cleaning...'
  for fn in os.listdir(_config.OUT_PLACE):
    if os.path.isdir(_config.OUT_PLACE + fn):
      print '  ', _config.OUT_PLACE + fn
      call('rm -rf ' + _config.OUT_PLACE + fn + '/*', 
        shell = True)
  print 'Done.'

def main():
  if _config.CLEAN == True:
    clean()

  if _config.CLEAN == 'ask':
    ans = raw_input('''
      Clean up output from intermediate steps in /out/? 
      Warning: You won\'t be able to recover data cleaned this way.
      Final results in /results/ are not touched.\n 
      [y/n]
      ''')
    if ans == 'y':
      clean()
  return


if __name__ == '__main__':
  main()