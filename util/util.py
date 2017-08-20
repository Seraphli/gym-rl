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


def init_logger(name, path=None):
    """Initialize a logger with certain name
    
    Args:
        name (str): logger name 
        path (str): optional, specify which folder path 
            the log file will be stored, for example
            '/tmp/log/DQN/Pong'

    Returns:
        logging.Logger: logger instance
    """
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

    if path:
        path += '/' + name + '.log'
    else:
        path = get_path('log') + '/' + name + '.log'
    rf = logging.handlers.RotatingFileHandler(path, maxBytes=5 * 1024 * 1024, backupCount=5)
    rf.setLevel(logging.DEBUG)
    rf.setFormatter(nformatter)

    logger.addHandler(ch)
    logger.addHandler(rf)
    return logger


def get_logger(algorithm, env, env_type):
    import datetime
    path = get_path('log/' + algorithm
                    + '/' + env + '-' + env_type
                    + '/' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    global main_logger, env_logger
    main_logger = init_logger('main', path)
    env_logger = init_logger('env', path)


def load_config(name):
    import yaml
    with open(get_path('config') + '/' + name) as f:
        return yaml.load(f)


def pretty_num(num, bit=False):
    mags = [('', 0, 0), ('K', 3, 10), ('M', 6, 20), ('G', 9, 30), ('T', 12, 40)]
    for unit, mag, mag_bit in mags[::-1]:
        if (not bit and num > 10 ** mag) or (bit and num > 2 ** mag_bit):
            break
    if unit == '':
        return "{}".format(num)
    n = num / (2 ** mag_bit) if bit else num / (10 ** mag)
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


def boolean_flag(parser, name, default=False, help=None):
    """Add a boolean flag to argparse parser.
    Parameters
    ----------
    parser: argparse.Parser
        parser to add the flag to
    name: str
        --<name> will enable the flag, while --no-<name> will disable it
    default: bool or None
        default value of the flag
    help: str
        help string for the flag
    """
    dest = name.replace('-', '_')
    parser.add_argument("--" + name, action="store_true", default=default, dest=dest, help=help)
    parser.add_argument("--no-" + name, action="store_false", dest=dest)


class RecentAvg(object):
    def __init__(self, size=50):
        self._value = None
        self._values = []
        self._size = size
        self._current = 0

    def update(self, new_val):
        if len(self._values) == self._size:
            self._values[self._current] = new_val
        else:
            self._values.append(new_val)
        self._current = (self._current + 1) % self._size
        self._value = sum(self._values) / len(self._values)

    @property
    def value(self):
        return self._value


class Record(object):
    def __init__(self):
        self._info = []
        length = [30, 15]
        self.length = sum(length) + 9
        self.format = {'kv': "|| {:>" + str(length[0]) + "} | {:>" + str(length[1]) + "} ||",
                       'l': "|| {:^" + str(sum(length) + 3) + "} ||"}

    def clear(self):
        self._info = []

    def add_key_value(self, key, value):
        self._info.append(('kv', key, value))

    def add_line(self, line):
        self._info.append(('l', '', line))

    def dumps(self, indent=''):
        record = ["=" * self.length]
        for t, k, v in self._info:
            if t == 'kv':
                record.append(self.format['kv'].format(k, v))
            if t == 'l':
                record.append(self.format['l'].format(v))
        record.append("=" * self.length)
        s = indent + ('\n' + indent).join(record)
        self._info = []
        return s
