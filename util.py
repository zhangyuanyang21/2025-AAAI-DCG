import logging
import numpy as np
import math
import datetime
# def get_logger():
#     """Get logging."""
#     logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
#     logger = logging.getLogger()
#     logger.setLevel(logging.DEBUG)
#     formatter = logging.Formatter(
#         '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S')
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.DEBUG)
#     ch.setFormatter(formatter)
#     logger.addHandler(ch)
#     return logger

#logs_comhMMI_comhcloss
#logs_comhMMI
#logs_comhcloss
#logs_comhKLloss

def get_logger(config):
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    plt_name = str(config['dataset']) + ' ' + str(config['training']['missing_rate']).replace('.','') + ' ' + str(
        datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H-%M-%S'))
    fh = logging.FileHandler(
        './logs_T/' + str(config['dataset']) + ' ' + str(config['training']['missing_rate']).replace('.','') + ' ' + str(
            datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H-%M-%S')) + '.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, plt_name

def next_batch(X1, X2, X3, X4, batch_size):
    """Return data for next batch"""
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size)
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        batch_x3 = X3[start_idx: end_idx, ...]
        batch_x4 = X4[start_idx: end_idx, ...]

        yield (batch_x1, batch_x2, batch_x3, batch_x4, (i + 1))

# def next_batch(X1, X2, batch_size):
#     """Return data for next batch"""
#     tot = X1.shape[0]
#     total = math.ceil(tot / batch_size)
#
#     for i in range(int(total)):
#         start_idx = i * batch_size
#         end_idx = (i + 1) * batch_size
#
#         # 如果最后一个批次不足batch_size大小，则跳过该批次
#         if end_idx > tot:
#             break
#
#         batch_x1 = X1[start_idx: end_idx, ...]
#         batch_x2 = X2[start_idx: end_idx, ...]
#
#         yield (batch_x1, batch_x2, (i + 1))

# def next_batch(X1, X2, batch_size, shuffle=True):
#     """Return data for the next batch"""
#     tot = X1.shape[0]
#
#     if shuffle:
#         # 如果设置为True，每个时代重新洗牌数据
#         indices = np.arange(tot)
#         np.random.shuffle(indices)
#         X1 = X1[indices, ...]
#         X2 = X2[indices, ...]
#
#     total = math.ceil(tot / batch_size)
#
#     for i in range(int(total)):
#         start_idx = i * batch_size
#         end_idx = (i + 1) * batch_size
#
#         # 如果最后一个批次不足batch_size大小，则跳过该批次
#         if end_idx > tot:
#             break
#
#         batch_x1 = X1[start_idx: end_idx, ...]
#         batch_x2 = X2[start_idx: end_idx, ...]
#
#         yield (batch_x1, batch_x2, (i + 1))


# def cal_std(logger, *arg):
#     """Return the average and its std"""
#     if len(arg) == 3:
#         logger.info('ACC:'+ str(arg[0]))
#         logger.info('NMI:'+ str(arg[1]))
#         logger.info('ARI:'+ str(arg[2]))
#         output = """ ACC {:.2f} std {:.2f} NMI {:.2f} std {:.2f} ARI {:.2f} std {:.2f}""".format(np.mean(arg[0]) * 100,
#                                                                                                  np.std(arg[0]) * 100,
#                                                                                                  np.mean(arg[1]) * 100,
#                                                                                                  np.std(arg[1]) * 100,
#                                                                                                  np.mean(arg[2]) * 100,
#                                                                                                  np.std(arg[2]) * 100)
#     elif len(arg) == 1:
#         logger.info(arg)
#         output = """ACC {:.2f} std {:.2f}""".format(np.mean(arg) * 100, np.std(arg) * 100)
#     logger.info(output)
#
#     return

def cal_std(logger, *arg):
    """ print clustering results """
    if len(arg) == 3:
        logger.info(arg[0])
        logger.info(arg[1])
        logger.info(arg[2])
        output = """ 
                     ACC {:.2f} std {:.2f}
                     NMI {:.2f} std {:.2f} 
                     ARI {:.2f} std {:.2f}""".format(np.mean(arg[0]) * 100, np.std(arg[0]) * 100, np.mean(arg[1]) * 100,
                                                     np.std(arg[1]) * 100, np.mean(arg[2]) * 100, np.std(arg[2]) * 100)
        logger.info(output)
        output2 = str(round(np.mean(arg[0]) * 100, 2)) + ',' + str(round(np.std(arg[0]) * 100, 2)) + ';' + \
                  str(round(np.mean(arg[1]) * 100, 2)) + ',' + str(round(np.std(arg[1]) * 100, 2)) + ';' + \
                  str(round(np.mean(arg[2]) * 100, 2)) + ',' + str(round(np.std(arg[2]) * 100, 2)) + ';'
        logger.info(output2)
        return round(np.mean(arg[0]) * 100, 2), round(np.mean(arg[1]) * 100, 2), round(np.mean(arg[2]) * 100, 2)

    elif len(arg) == 1:
        logger.info(arg)
        output = """ACC {:.2f} std {:.2f}""".format(np.mean(arg) * 100, np.std(arg) * 100)
        logger.info(output)

def normalize(x):
    """Normalize"""
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x

def target_l2(q):
    return ((q ** 2).t() / (q ** 2).sum(1)).t()