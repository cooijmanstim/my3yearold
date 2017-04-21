import sys, os, subprocess
from collections import defaultdict
from natsort import natsorted
import scipy.misc
import numpy as np

logpath = sys.argv[1]
log = np.load(logpath)
outpath = "%s_extract" % logpath

frames = defaultdict(list)

for subpath in natsorted(log.keys()):
  if not subpath.endswith("_x"):
    continue

  xs = log[subpath]

  # save batch of images
  for i, x in enumerate(xs):
    pngpath = "%s_%i.png" % (os.path.join(outpath, subpath), i)

    whenever_i_want_to_do_something_in_python_i_feel_like_my_hands_are_full_and_i_need_to_first_put_everything_down_into_a_temporary_variable = os.path.dirname(pngpath)
    if not os.path.exists(whenever_i_want_to_do_something_in_python_i_feel_like_my_hands_are_full_and_i_need_to_first_put_everything_down_into_a_temporary_variable):
      os.makedirs(whenever_i_want_to_do_something_in_python_i_feel_like_my_hands_are_full_and_i_need_to_first_put_everything_down_into_a_temporary_variable)

    print pngpath
    scipy.misc.imsave(pngpath, x)
    frames[i].append(pngpath)

for key, paths in frames.items():
  gifpath = os.path.join(outpath, "%s.gif" % key)
  subprocess.check_call("convert -delay 10 -loop 0".split() +
                        paths +
                        "-delay 500".split() +
                        paths[-1:] +
                        [gifpath])

htmlpath = os.path.join(outpath, "index.html")
with open(htmlpath, "w") as htmlfile:
  htmlfile.write("\n".join("<img src=%s.gif>" % key for key in sorted(frames.keys())))
