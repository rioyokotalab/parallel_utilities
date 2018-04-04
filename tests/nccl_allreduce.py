import chainer
import chainer.testing
import chainer.testing.attr
import unittest

import chainermn
import mpi4py.MPI

class ExampleModel(chainer.Chain):
    def __init__(self, n_in=3, n_units=5, n_out=2):
        super(ExampleModel, self).__init__(
            l1=chainer.links.Linear(n_in, n_units, nobias=True),
            bn1=chainer.links.BatchNormalization(n_units),
            l2=chainer.links.Linear(n_units, n_units, nobias=True),
            bn2=chainer.links.BatchNormalization(n_units),
            l3=chainer.links.Linear(n_units, n_out),
        )


class TestAllreducePersistent(unittest.TestCase):

    def _test(self, comm, model):
        rank = comm.rank
        #model.bn1.avg_mean.fill(rank * 1)
        #model.bn2.avg_mean.fill(rank * 2)
        #model.bn1.avg_var.fill(rank * 3)
        #model.bn2.avg_var.fill(rank * 4)

        #allreduce_persistent = \
        #    chainermn.extensions.AllreducePersistent(model, comm)
        #allreduce_persistent()

        #avg_rank = (comm.size - 1) / 2.0
        #chainer.testing.assert_allclose(model.bn1.avg_mean, avg_rank * 1)
        #chainer.testing.assert_allclose(model.bn2.avg_mean, avg_rank * 2)
        #chainer.testing.assert_allclose(model.bn1.avg_var, avg_rank * 3)
        #chainer.testing.assert_allclose(model.bn2.avg_var, avg_rank * 4)
        #print("model.bn1.avg_mean:{}".format(model.bn1.avg_mean))
        #print("avg_rank:{}".format(avg_rank))
        print("rank:{}".format(comm.rank))

    def test_allreduce_persistent_gpu(self):
        mpi_comm = mpi4py.MPI.COMM_WORLD
        communicator_name = 'pure_nccl'
        wcomm = chainermn.create_communicator(
            communicator_name=communicator_name, mpi_comm=mpi_comm)

        color_value = wcomm.rank//2
        key_value = wcomm.rank%2
        print("wcomm.rank : {} / color_value : {} / key_value :  {}".format(wcomm.rank,color_value,key_value))

        #scomm = wcomm.split(color=color_value, key=key_value)
        device = wcomm.intra_rank
        #device = wcomm.intra_rank*2+scomm.intra_rank
        chainer.cuda.get_device(device).use()
        #print("wcomm.rank : {} / scomm.rank : {}".format(wcomm.rank,scomm.rank))
        # print("wcomm.intra_rank : {} / scomm.intra_rank : {}".format(wcomm.intra_rank,scomm.intra_rank))
        # print("wcomm.mpi_comm.rank : {} / scomm.mpi_comm.rank : {}".format(wcomm.mpi_comm.rank,scomm.mpi_comm.rank))
        model = ExampleModel()
        model.to_gpu()
        self._test(wcomm, model)

if __name__ == '__main__':
    unittest.main()
