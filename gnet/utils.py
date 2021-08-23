from logging import getLogger, Formatter, StreamHandler

def get_logger(name=''):
    logger = getLogger(name) 

    # formatter
    fmr = Formatter('%(asctime)s - %(filename)s:%(lineno)s - %(levelname)s:%(message)s')

    # stream handler
    ch = StreamHandler()
    ch.setFormatter(fmr)

    logger.addHandler(ch)

    return logger