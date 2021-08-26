from logging import INFO, getLogger, Formatter, StreamHandler

__all__ = ['get_logger']
__logger = None

def get_logger():
    global __logger
    if __logger is not None:
        return __logger
    logger = getLogger()
    logger.setLevel(INFO)

    # formatter
    fmr = _ColoredFormatter('G2Net: %(filename)s:%(lineno)s - %(levelname)s:  %(message)s')

    # stream handler
    ch = StreamHandler()
    ch.setLevel(INFO)
    ch.setFormatter(fmr)

    logger.addHandler(ch)
    __logger = logger
    
    return __logger


BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
#The background is set with 40 plus the number of the color, and the foreground with 30

#These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"

_COLORS = {
    'WARNING': YELLOW,
    'INFO': WHITE,
    'DEBUG': BLUE,
    'CRITICAL': YELLOW,
    'ERROR': RED
}

class _ColoredFormatter(Formatter):
    def __init__(self, msg, use_color = True):
        Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in _COLORS:
            levelname_color = COLOR_SEQ % (30 + _COLORS[levelname]) + levelname + RESET_SEQ
            record.levelname = levelname_color
        return Formatter.format(self, record)