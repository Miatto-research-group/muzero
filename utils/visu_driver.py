from visualisation_tool import *
from copy import deepcopy

y = [10,20,30,40,50,60,70,80,90, 100]
rev_y = deepcopy(y)
rev_y.reverse( )
x = [1,2,3,4,5,6,7,8,9,10]
ys = [y, rev_y]
labs = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
do_plot_2D(x, ys, labs, "my_file", "Some plot title", gnplt=False, col_offset=3, x_lab="timestep", y_lab="perf")