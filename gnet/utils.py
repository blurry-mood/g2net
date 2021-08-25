from logging import INFO, getLogger, Formatter, StreamHandler

def get_logger(name=''):
    logger = getLogger(name)
    logger.setLevel(INFO)

    # formatter
    fmr = Formatter('%(filename)s:%(lineno)s - %(levelname)s:  %(message)s')

    # stream handler
    ch = StreamHandler()
    ch.setLevel(INFO)
    ch.setFormatter(fmr)

    logger.addHandler(ch)

    return logger