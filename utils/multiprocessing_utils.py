import copy

import torch
import torch.multiprocessing as mp

#模拟队列的类，它提供了队列的基本接口，但实际上并不执行任何队列操作
class FakeQueue:
    def put(self, arg): #接受一个参数arg，但实际上并不存储这个参数，而是直接删除它
        del arg

    def get_nowait(self): #模拟了尝试从队列中获取元素的操作，但由于FakeQueue不存储任何元素，所以它直接抛出mp.queues.Empty异常，表示队列为空
        raise mp.queues.Empty

    def qsize(self): #返回队列中元素的数量，对于FakeQueue，这个数量始终为0
        return 0

    def empty(self): #检查队列是否为空，对于FakeQueue，这个方法始终返回True，表示队列为空
        return True


def clone_obj(obj): #深度复制一个对象，并进行处理
    clone_obj = copy.deepcopy(obj) #使用copy.deepcopy(obj)深度复制输入的对象obj
    for attr in clone_obj.__dict__.keys(): #遍历复制后对象的所有属性
        # check if its a property
        if hasattr(clone_obj.__class__, attr) and isinstance( #对于定义为类属性的property，不进行处理
            getattr(clone_obj.__class__, attr), property
        ):
            continue
        if isinstance(getattr(clone_obj, attr), torch.Tensor): #如果属性是一个torch.Tensor，则使用detach().clone()方法创建该张量的一个脱离原始计算图的副本，并将这个副本设置回复制后的对象的相应属性中
            setattr(clone_obj, attr, getattr(clone_obj, attr).detach().clone())
    return clone_obj
