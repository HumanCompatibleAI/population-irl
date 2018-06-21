class SeqComm:
    '''Dummy MPI communicator that is actually sequential.'''
    def __init__(self):
        pass

    @staticmethod
    def Get_size():
        return 1

    @staticmethod
    def Get_rank():
        return 0

    @staticmethod
    def Allreduce(sendbuf, recvbuf, op):
        recvbuf[:] = sendbuf[:]

    @staticmethod
    def Bcast(buf, root):
        pass

    @staticmethod
    def allgather(sendobj):
        return [sendobj]

class MPI:
    '''Dummy MPI interface that is actually sequential.'''
    COMM_WORLD = SeqComm()
    SUM        = 0

__all__ = ['MPI']

