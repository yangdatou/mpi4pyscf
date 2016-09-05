#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''Density expansion on plane waves'''

import time
import copy
import numpy

from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc.gto import pseudo
from pyscf.pbc.df import pwdf
from pyscf.pbc.df import ft_ao

from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi
from mpi4pyscf.pbc.df import pwdf_jk
#from mpi4pyscf.pbc.df import pwdf_ao2mo

comm = mpi.comm
rank = mpi.rank


def init_PWDF(cell, kpts=numpy.zeros((1,3))):
    mydf = mpi.pool.apply(_init_PWDF_wrap, [cell, kpts], [cell.dumps(), kpts])
    return mydf
def _init_PWDF_wrap(args):
    from mpi4pyscf.pbc.df import pwdf
    cell, kpts = args
    if pwdf.rank > 0:
        cell = pwdf.gto.loads(cell)
        cell.verbose = 0
    return pwdf.mpi.register_for(pwdf.PWDF(cell, kpts))

def get_nuc(mydf, kpts=None):
    args = (mydf._reg_keys, kpts)
    return mpi.pool.apply(_get_nuc_wrap, args, args)
def _get_nuc_wrap(args):
    from mpi4pyscf.pbc.df import pwdf
    return pwdf._get_nuc(*args)
def _get_nuc(reg_keys, kpts):
    mydf = pwdf_jk._load_df(reg_keys)
    vne = lib.asarray(pwdf.get_nuc(mydf, kpts))
    vne = mpi.reduce(vne)
    return vne

def get_pp(mydf, kpts=None):
    args = (mydf._reg_keys, kpts)
    return mpi.pool.apply(_get_pp_wrap, args, args)
def _get_pp_wrap(args):
    from mpi4pyscf.pbc.df import pwdf
    return pwdf._get_pp(*args)
def _get_pp(reg_keys, kpts):
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

    mydf = pwdf_jk._load_df(reg_keys)
    vpp = pwdf.get_pp_loc_part1(mydf, mydf.cell, kpts_lst)
    vpp = mpi.reduce(lib.asarray(vpp))

    if rank == 0:
        vloc2 = pseudo.pp_int.get_pp_loc_part2(mydf.cell, kpts_lst)
        vppnl = pseudo.pp_int.get_pp_nl(mydf.cell, kpts_lst)
        for k in range(len(kpts_lst)):
            vpp[k] += numpy.asarray(vppnl[k] + vloc2[k], dtype=vpp.dtype)

        if kpts is None or numpy.shape(kpts) == (3,):
            vpp = vpp[0]
        return vpp


class PWDF(pwdf.PWDF):
    def __enter__(self):
        return self
    def __exit__(self):
        self.close()
    def close(self):
        self._reg_keys = mpi.del_registry(self._reg_keys)

    def mpi_pw_loop(self, cell, gs=None, kpti_kptj=None, shls_slice=None,
                    max_memory=2000):
        '''Plane wave part'''
        if gs is None:
            gs = self.gs
        if kpti_kptj is None:
            kpti = kptj = numpy.zeros(3)
        else:
            kpti, kptj = kpti_kptj

        nao = cell.nao_nr()
        gxyz = lib.cartesian_prod((numpy.append(range(gs[0]+1), range(-gs[0],0)),
                                   numpy.append(range(gs[1]+1), range(-gs[1],0)),
                                   numpy.append(range(gs[2]+1), range(-gs[2],0))))
        invh = numpy.linalg.inv(cell._h)
        Gv = 2*numpy.pi * numpy.dot(gxyz, invh)
        ngs = gxyz.shape[0]

# Theoretically, hermitian symmetry can be also found for kpti == kptj:
#       f_ji(G) = \int f_ji exp(-iGr) = \int f_ij^* exp(-iGr) = [f_ij(-G)]^*
# The hermi operation needs reordering the axis-0.  It is inefficient
        if gamma_point(kpti) and gamma_point(kptj):
            aosym = 's1hermi'
        else:
            aosym = 's1'

        blksize = min(max(16, int(max_memory*1e6*.7/16/nao**2)), 16384)
        sublk = max(16, int(blksize//4))
        pqkRbuf = numpy.empty(nao*nao*sublk)
        pqkIbuf = numpy.empty(nao*nao*sublk)

        for p0, p1 in self.mpi_prange(0, ngs, blksize):
            aoao = ft_ao.ft_aopair(cell, Gv[p0:p1], shls_slice, aosym, invh,
                                   gxyz[p0:p1], gs, (kpti, kptj))
            for i0, i1 in lib.prange(0, p1-p0, sublk):
                nG = i1 - i0
                pqkR = numpy.ndarray((nao,nao,nG), buffer=pqkRbuf)
                pqkI = numpy.ndarray((nao,nao,nG), buffer=pqkIbuf)
                pqkR[:] = aoao[i0:i1].real.transpose(1,2,0)
                pqkI[:] = aoao[i0:i1].imag.transpose(1,2,0)
                yield (pqkR.reshape(-1,nG),
                       pqkI.reshape(-1,nG), p0+i0, p0+i1)

    def mpi_ft_loop(self, cell, gs=None, kpt=numpy.zeros(3),
                    kpts=None, shls_slice=None, max_memory=4000):
        '''
        Fourier transform iterator for all kpti which satisfy  kpt = kpts - kpti
        '''
        if gs is None: gs = self.gs
        if kpts is None:
            assert(gamma_point(kpt))
            kpts = self.kpts
        kpts = numpy.asarray(kpts)
        nkpts = len(kpts)

        nao = cell.nao_nr()
        gxyz = lib.cartesian_prod((numpy.append(range(gs[0]+1), range(-gs[0],0)),
                                   numpy.append(range(gs[1]+1), range(-gs[1],0)),
                                   numpy.append(range(gs[2]+1), range(-gs[2],0))))
        invh = numpy.linalg.inv(cell._h)
        Gv = 2*numpy.pi * numpy.dot(gxyz, invh)
        ngs = gxyz.shape[0]

# Theoretically, hermitian symmetry can be also found for kpti == kptj:
#       f_ji(G) = \int f_ji exp(-iGr) = \int f_ij^* exp(-iGr) = [f_ij(-G)]^*
# The hermi operation needs reordering the axis-0.  It is inefficient
        if gamma_point(kpt) and gamma_point(kpts):
            aosym = 's1hermi'
        else:
            aosym = 's1'

        blksize = min(max(16, int(max_memory*1e6*.9/(nao**2*(nkpts+1)*16))), 16384)
        buf = [numpy.zeros(nao*nao*blksize, dtype=numpy.complex128)
               for k in range(nkpts)]
        pqkRbuf = numpy.empty(nao*nao*blksize)
        pqkIbuf = numpy.empty(nao*nao*blksize)

        for p0, p1 in self.mpi_prange(0, ngs, blksize):
            ft_ao._ft_aopair_kpts(cell, Gv[p0:p1], shls_slice, aosym, invh,
                                  gxyz[p0:p1], gs, kpt, kpts, out=buf)
            nG = p1 - p0
            for k in range(nkpts):
                aoao = numpy.ndarray((nG,nao,nao), dtype=numpy.complex128,
                                     order='F', buffer=buf[k])
                pqkR = numpy.ndarray((nao,nao,nG), buffer=pqkRbuf)
                pqkI = numpy.ndarray((nao,nao,nG), buffer=pqkIbuf)
                pqkR[:] = aoao.real.transpose(1,2,0)
                pqkI[:] = aoao.imag.transpose(1,2,0)
                yield (k, pqkR.reshape(-1,nG), pqkI.reshape(-1,nG), p0, p1)
                aoao[:] = 0  # == buf[k][:] = 0

    def mpi_prange(self, start, stop, step):
        mpi_size = mpi.pool.size
        step = min(step, (stop-start+mpi_size-1)//mpi_size)
        task_lst = [(p0,p1) for p0, p1 in lib.prange(start, stop, step)]
        return mpi.static_partition(task_lst)

    get_nuc = get_nuc
    get_pp = get_pp

    def get_jk(self, dm, hermi=1, kpts=None, kpt_band=None,
               with_j=True, with_k=True, exxdiv='ewald'):
        '''Gamma-point calculation by default'''
        if kpts is None:
            if numpy.all(self.kpts == 0):
                kpts = numpy.zeros(3)
            else:
                kpts = self.kpts
        else:
            kpts = numpy.asarray(kpts)

        # Use DF object to mimic KRHF/KUHF object in function get_coulG
        self.exxdiv = exxdiv

        if kpts.shape == (3,):
            return pwdf_jk.get_jk(self, dm, hermi, kpts, kpt_band, with_j, with_k)

        vj = vk = None
        if with_k:
            vk = pwdf_jk.get_k_kpts(self, dm, hermi, kpts, kpt_band)
        if with_j:
            vj = pwdf_jk.get_j_kpts(self, dm, hermi, kpts, kpt_band)
        return vj, vk

def is_zero(kpt):
    return kpt is None or abs(kpt).sum() < 1e-9
gamma_point = is_zero


if __name__ == '__main__':
    # run with mpirun -n
    from pyscf.pbc import gto as pgto
    from mpi4pyscf.pbc import df
    cell = pgto.Cell()
    cell.atom = 'He 1. .5 .5; C .1 1.3 2.1'
    cell.basis = {'He': [(0, (2.5, 1)), (0, (1., 1))],
                  'C' :'gth-szv',}
    cell.pseudo = {'C':'gth-pade'}
    cell.h = numpy.eye(3) * 2.5
    cell.gs = [5] * 3
    cell.build()
    numpy.random.seed(19)
    kpts = numpy.random.random((5,3))

    mydf = df.PWDF(cell)
    v = mydf.get_nuc()
    print(v.shape)
    v = mydf.get_pp(kpts)
    print(v.shape)

    cell = pgto.M(atom='He 0 0 0; He 0 0 1', h=numpy.eye(3)*4, gs=[5]*3)
    mydf = df.PWDF(cell)
    nao = cell.nao_nr()
    dm = numpy.ones((nao,nao))
    vj, vk = mydf.get_jk(dm)
    print(vj.shape)
    print(vk.shape)

    dm_kpts = [dm]*5
    vj, vk = mydf.get_jk(dm_kpts, kpts=kpts)
    print(vj.shape)
    print(vk.shape)

    mydf.close()

