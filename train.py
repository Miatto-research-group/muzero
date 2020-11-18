from muzero import Muzero
from environments import TicTacToe
from tqdm import trange
import torch.multiprocessing as mp

from loguru import logger
logger.add("logfile.log")




if __name__=='__main__':

    muzero = Muzero(TicTacToe)
    muzero.network.representation.share_memory()
    muzero.network.prediction.share_memory()
    muzero.network.dynamics.share_memory()
    num_processes = 4
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=M.train)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    M = Muzero(TicTacToe)
    M.fill_replay_buffer(games=10)
    logger.info(win_draw_lose(M.REPLAY_BUFFER))

    for _ in range(20):
        M.optimize_step()

    M.fill_replay_buffer(games=10)
    logger.info(win_draw_lose(M.REPLAY_BUFFER))

    


    # TODO save trained model