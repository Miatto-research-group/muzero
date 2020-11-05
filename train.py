from muzero import Muzero
from environments import TicTacToe
from tqdm import trange

from loguru import logger
logger.add("logfile.log")


def win_draw_lose(buffer):
    win = 0
    draw = 0
    lose = 0
    for _,_,_,r in buffer:
        win += int(r[-1]==1)
        draw += int(r[-1]==0)
        lose += int(r[-1]==-1)
    return f'win: {win} | draw: {draw} | lose: {lose}'


if __name__=='__main__':

    M = Muzero(TicTacToe)
    M.fill_replay_buffer(games=10)
    logger.info(win_draw_lose(M.REPLAY_BUFFER))

    for _ in range(20):
        M.optimize_step()

    M.fill_replay_buffer(games=10)
    logger.info(win_draw_lose(M.REPLAY_BUFFER))

    


    # TODO save trained model