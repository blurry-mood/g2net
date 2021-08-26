from logging import DEBUG, INFO, getLogger, Formatter, StreamHandler

__all__ = ['get_logger']

def get_logger():
    logger = getLogger('G2Net')
    logger.setLevel(DEBUG)

    # formatter

    # stream handler
    if not logger.hasHandlers():
        fmr = _ColoredFormatter('%(name)s: %(filename)s:%(lineno)s - %(levelname)s:  %(message)s')
        ch = StreamHandler()
        ch.setLevel(DEBUG)
        ch.setFormatter(fmr)

        logger.addHandler(ch)
    
    return logger


BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
#The background is set with 40 plus the number of the color, and the foreground with 30

#These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[;%dm"
BOLD_COLOR_SEQ = "\033[1;%dm"

_COLORS = {
    'WARNING': YELLOW,
    'INFO': GREEN,
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

        # name
        name = record.name
        if self.use_color:
            name_color = BOLD_COLOR_SEQ % (30 + RED) + name + RESET_SEQ
            record.name = name_color
        return Formatter.format(self, record)