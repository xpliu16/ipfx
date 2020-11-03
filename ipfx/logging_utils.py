import os
import logging


def configure_logger(cell_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []
    file_handler = logging.FileHandler(filename=os.path.join(cell_dir,"log.txt"), mode='w')
    logger.addHandler(file_handler)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    logger.addHandler(stream_handler)


def log_pretty_header(header, level=1, top_line_break=True, bottom_line_break=True):
    """
    Decorate logging message to make logging output more human readable

    Parameters
    ----------
    header: str
        header message
    level: int
        1 or 2 as in markdown
    top_line_break: bool (True)
        add a blank line at the top
    bottom_line_break: bool (True)
        add a blank line at the bottom
    """

    if top_line_break:
        logging.info("  ")

    header = "***** ***** ***** " + header + " ***** ***** *****"
    logging.info(header)

    if level ==1:
        logging.info("="*len(header))
    elif level == 2:
        logging.info("-"*len(header))

    if bottom_line_break:
        logging.info("  ")

