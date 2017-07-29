import logging


def get_path(name):
    import os
    directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, name))
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"
COLORS = {
    'WARNING': YELLOW,
    'INFO': WHITE,
    'DEBUG': BLUE,
    'CRITICAL': YELLOW,
    'ERROR': RED,
    'RED': RED,
    'GREEN': GREEN,
    'YELLOW': YELLOW,
    'BLUE': BLUE,
    'MAGENTA': MAGENTA,
    'CYAN': CYAN,
    'WHITE': WHITE,
}


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg):
        logging.Formatter.__init__(self, msg)

    def format(self, record):
        levelname = record.levelname
        if levelname in COLORS:
            levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
            record.levelname = levelname_color
        message = logging.Formatter.format(self, record)
        message = message.replace("$RESET", RESET_SEQ) \
            .replace("$BOLD", BOLD_SEQ)
        for k, v in COLORS.items():
            message = message.replace("$" + k, COLOR_SEQ % (v + 30)) \
                .replace("$BG" + k, COLOR_SEQ % (v + 40)) \
                .replace("$BG-" + k, COLOR_SEQ % (v + 40))
        return message + RESET_SEQ


def init_logger(name):
    """
    Initialize a logger with certain name
    :param name: logger name
    :return: logger
    :rtype: logging.Logger
    """
    import logging
    import logging.handlers
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = 0
    _nf = ['[%(asctime)s]',
           '[%(name)s]',
           '[%(filename)20s:%(funcName)15s:%(lineno)5d]',
           '[%(levelname)s]',
           ' %(message)s']
    _cf = ['$GREEN[%(asctime)s]$RESET',
           '[%(name)s]',
           '$BLUE[%(filename)20s:%(funcName)15s:%(lineno)5d]$RESET',
           '[%(levelname)s]',
           ' $CYAN%(message)s$RESET']
    nformatter = logging.Formatter('-'.join(_nf))
    cformatter = ColoredFormatter('-'.join(_cf))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(cformatter)

    rf = logging.handlers.RotatingFileHandler(get_path('log') + '/' + name + '.log',
                                              maxBytes=1 * 1024 * 1024,
                                              backupCount=5)
    rf.setLevel(logging.DEBUG)
    rf.setFormatter(nformatter)

    logger.addHandler(ch)
    logger.addHandler(rf)
    return logger


main_logger = init_logger('main')


def load_config(name):
    import yaml
    with open(get_path('config') + '/' + name) as f:
        return yaml.load(f)


def pretty_num(num):
    mags = [('', 0), ('k', 3), ('m', 6), ('g', 9), ('t', 12)]
    for unit, mag in mags[::-1]:
        if num > 10 ** mag:
            break
    if unit == '':
        return "{}".format(num)
    n = num / (10 ** mag)
    return "{:.2f} {}".format(n, unit)


def pretty_eta(seconds_left):
    """Print the number of seconds in human readable format.
    Examples:
    2 days
    2 hours and 37 minutes
    less than a minute
    Paramters
    ---------
    seconds_left: int
        Number of seconds to be converted to the ETA
    Returns
    -------
    eta: str
        String representing the pretty ETA.
    """
    minutes_left = seconds_left // 60
    seconds_left %= 60
    hours_left = minutes_left // 60
    minutes_left %= 60
    days_left = hours_left // 24
    hours_left %= 24

    def helper(cnt, name):
        return "{} {}{}".format(str(cnt), name, ('s' if cnt > 1 else ''))

    if days_left > 0:
        msg = helper(days_left, 'day')
        if hours_left > 0:
            msg += ' and ' + helper(hours_left, 'hour')
        return msg
    if hours_left > 0:
        msg = helper(hours_left, 'hour')
        if minutes_left > 0:
            msg += ' and ' + helper(minutes_left, 'minute')
        return msg
    if minutes_left > 0:
        return helper(minutes_left, 'minute')
    return 'less than a minute'


class RunningAvg(object):
    def __init__(self, gamma, init_value=None):
        """Keep a running estimate of a quantity. This is a bit like mean
        but more sensitive to recent changes.
        Parameters
        ----------
        gamma: float
            Must be between 0 and 1, where 0 is the most sensitive to recent
            changes.
        init_value: float or None
            Initial value of the estimate. If None, it will be set on the first update.
        """
        self._value = init_value
        self._gamma = gamma

    def update(self, new_val):
        """Update the estimate.
        Parameters
        ----------
        new_val: float
            new observated value of estimated quantity.
        """
        if self._value is None:
            self._value = new_val
        else:
            self._value = self._gamma * self._value + (1.0 - self._gamma) * new_val

    def __float__(self):
        """Get the current estimate"""
        return self._value


class Record(object):
    def __init__(self):
        self._info = []
        self.format = {'kv': "|| {:>30} | {:>15} ||", 'l': "|| {:^48} ||"}

    def clear(self):
        self._info = []

    def add_key_value(self, key, value):
        self._info.append(('kv', key, value))

    def add_line(self, line):
        self._info.append(('l', '', line))

    def dumps(self, indent=''):
        record = ["=" * 54]
        for t, k, v in self._info:
            if t == 'kv':
                record.append(self.format['kv'].format(k, v))
            if t == 'l':
                record.append(self.format['l'].format(v))
        record.append("=" * 54)
        s = indent + ('\n' + indent).join(record)
        self._info = []
        return s
