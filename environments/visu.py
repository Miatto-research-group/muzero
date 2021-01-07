###############################################################################
###                                   IMPORTS                               ###
###############################################################################
import matplotlib.pyplot as plt
import os
import numpy as np


###############################################################################
###                            REPRODUCIBILITY                              ###
###############################################################################
np.random.seed(42)
ub = 2**32 - 1
seeds =  np.random.randint(ub, size=10)



###############################################################################
###                           GENERATING PLOTS                              ###
###############################################################################
def do_plot_2D(x, y, labs, f_name, tit, logx=False, logy=False, gnplt=True, col_offset=0, x_lab="X", y_lab="Y"):
    if gnplt:
        plt.style.use('ggplot')

    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'plots/')
    file_name = "plot_" + f_name + ".png"

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    plt.xlabel(x_lab)
    plt.ylabel(y_lab)

    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")

    for i in range(len(y)):
        curr_y = y[i]
        curr_lab = labs[i]
        col = 'C{}'.format(i+col_offset)
        plt.plot(x, curr_y, label=curr_lab, color=col)

    plt.legend()
    plt.title(tit)
    plt.savefig(results_dir + file_name)
    plt.close()



###############################################################################
###                         VISUALISING UNITARIES                           ###
###############################################################################
def to_matrix(some_tensor:np.array) -> np.matrix:
    res = None
    if (some_tensor.shape == (2, 2)):
        res = some_tensor
    elif (some_tensor.shape == (2, 2, 2, 2)):
        res = np.einsum('abcd->dacb', some_tensor)
        res = np.reshape(res, (4, 4))
    elif (some_tensor.shape == (2, 2, 2, 2, 2, 2)):
        res = np.einsum('abcdef->acebdf', some_tensor)
        res = np.reshape(res, (8, 8))
    else:
        raise ValueError('Unsupported system dimension for matrix visualisation')
    return np.matrix(res)