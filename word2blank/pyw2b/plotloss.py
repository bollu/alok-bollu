#!/usr/bin/env python3
# Plot average loss.
from bokeh.plotting import figure, output_file, show, save
import os, errno
import socket
import getpass
import glob

def mkdirs(newdir):
    try:
        os.makedirs(newdir)
    except OSError as err:
        # Reraise the error unless it's about an already existing directory 
        if err.errno != errno.EEXIST or not os.path.isdir(newdir): 
            raise


def scp_command(path):
    """make the command to scp ABSOLUTE  PATH path"""
    hostname = socket.gethostname()
    username = getpass.getuser()
    # return "rsync -r -avz --progress %s@%s:%s" % (username, hostname, path)
    return "rsync -r -avz --progress %s@ada.iiit.ac.in:%s ." % (username, path)


def extract_avg_per_elem(raw):
    """raw: all text in one single string"""
    lines = raw.split("\n")
    lines = list(filter(lambda l: l.startswith("LOSSES"), lines))
    return list(map (lambda l: float(l.split("avg per elements(")[1].split("):")[1]), lines))


LOSSES_RELATIVE_FILEPATH = "plots/losses.html"

if __name__ == "__main__":
    mkdirs("plots")
    # bokeh file to write to
    output_file(LOSSES_RELATIVE_FILEPATH)

    avgs = {}
    NAMES = glob.glob("*.log")
    for name in NAMES
        with open(path, "r") as f:
            avgs[path] = extract_avg_per_elem(f.read())


    maxlen = max(map (lambda ls: len(ls), avgs.values()))
    x = list(range(maxlen))

    p = figure(title="losses", x_axis_label='tick', y_axis_label='loss')
    for name in avgs:
        p.line(x, avgs[name], legend=name)

    save(p)

    print("run command to copy:\n%s" % 
	  (scp_command(os.path.join(os.getcwd(), "plots"))))
