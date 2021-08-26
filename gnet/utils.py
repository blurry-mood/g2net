from logging import INFO, getLogger, Formatter, StreamHandler

__all__ = ['get_logger']

def get_logger():
    logger = getLogger()
    logger.setLevel(INFO)

    # formatter
    fmr = Formatter('G2Net: %(filename)s:%(lineno)s - %(levelname)s:  %(message)s')

    # stream handler
    ch = StreamHandler()
    ch.setLevel(INFO)
    ch.setFormatter(fmr)

    logger.addHandler(ch)

    return logger