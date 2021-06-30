import _pickle as cPickle


def save_pkl(obj, path):
  with open(path, 'wb') as f:  # with语句用于处理异常状况
    cPickle.dump(obj, f)  # 以文本的形式序列化对象，并将结果数据流写入到文件对象中
    # print("  [*] save %s" % path)


def load_pkl(path):
  with open(path, 'rb') as f:
    obj = cPickle.load(f)  # 反序列化对象，将文件中的数据解析为一个Python对象。
    # print("  [*] load %s" % path)
    return obj