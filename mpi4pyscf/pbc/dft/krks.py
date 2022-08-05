#!/usr/bin/env python

import platform
import time
import numpy
from pyscf import lib

from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import rks
from pyscf.pbc.dft import multigrid

from mpi4pyscf.lib import logger
from mpi4pyscf.scf import hf as mpi_hf
from mpi4pyscf.tools import mpi

comm = mpi.comm
rank = mpi.rank

@mpi.parallel_call(skip_args=[1, 2, 3, 4], skip_kwargs=['dm_last', 'vhf_last'])
def get_veff_mpi(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
                     kpts=None, kpts_band=None):
    t0 = (logger.process_clock(), logger.perf_counter())
    ks.unpack_(comm.bcast(ks.pack()))
    cell = ks.cell
    kpts = ks.kpts
    ni = ks._numint

    if ks.nlc != '':
        raise NotImplementedError

    omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=cell.spin)
    hybrid = abs(hyb) > 1e-10 or abs(alpha) > 1e-10

    # Broadcast the large input arrays here.
    if any(comm.allgather(dm is mpi.Message.SkippedArg)):
        if rank == 0 and dm is None:
            dm = ks.make_rdm1()
        dm = mpi.bcast_tagged_array(dm)

    ground_state = (isinstance(dm, numpy.ndarray) and dm.ndim == 3 and kpts_band is None)

    if not hybrid and isinstance(ks.with_df, multigrid.MultiGridFFTDF):
        logger.warn(ks, "MultiGridFFTDF is not supported for MPI")
        n, exc, vxc = multigrid.nr_rks(ks.with_df, ks.xc, dm, hermi,
                                       kpts, kpts_band,
                                       with_j=True, return_j=False)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)
        return vxc

    if ks.grids.non0tab is None:
        _setup_becke_grids(ks, dm, ground_state)
        t0 = logger.timer(ks, 'setting up grids', *t0)

    if hermi == 2:  # because rho = 0
        n, exc, vxc = 0, 0, 0
    else:
        n, exc, vxc = ni.nr_rks(cell, ks.grids, ks.xc, dm, 
                                      0, kpts, kpts_band, 
                                      max_memory=ks.max_memory,
                                      verbose=ks.verbose)
        n = comm.allreduce(n)
        exc = comm.allreduce(exc)
        vxc = mpi.reduce(vxc)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)
    
    weight = 1./len(kpts)

    if not hybrid:
        vj = ks.get_j(cell, dm, hermi, kpts, kpts_band)
        vxc += vj
    else:
        if getattr(ks.with_df, '_j_only', False):  # for GDF and MDF
            ks.with_df._j_only = False
        vj, vk = ks.get_jk(cell, dm, hermi, kpts, kpts_band)
        vk *= hyb
        if abs(omega) > 1e-10:
            vklr = ks.get_k(cell, dm, hermi, kpts, kpts_band, omega=omega)
            vklr *= (alpha - hyb)
            vk += vklr
        vxc += vj - vk * .5

        if ground_state:
            exc -= numpy.einsum('Kij,Kji', dm, vk).real * .5 * .5 * weight

    if ground_state:
        ecoul = numpy.einsum('Kij,Kji', dm, vj).real * .5 * weight
    else:
        ecoul = None

    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=None, vk=None)
    return vxc

def _setup_uniform_grids():
    pass

def _setup_becke_grids(ks, dm, ground_state):
    cell  = ks.cell
    grids = ks.grids

    if rank == 0:
        grids.build(with_non0tab=False)
        ngrids = comm.bcast(grids.weights.size)
        grids.coords = numpy.array_split(grids.coords, mpi.pool.size)
        grids.weights = numpy.array_split(grids.weights, mpi.pool.size)
    else:
        ngrids = comm.bcast(None)
    grids.coords = mpi.scatter(grids.coords)
    grids.weights = mpi.scatter(grids.weights)

    if (isinstance(ks.grids, gen_grid.BeckeGrids) 
        and ks.small_rho_cutoff > 1e-20 
        and ground_state):

        rho = ks._numint.get_rho(cell, dm, grids, ks.max_memory)
        n = comm.allreduce(numpy.dot(rho, grids.weights))

        if abs(n-cell.nelectron) < rks.NELEC_ERROR_TOL*n:
            rw = rho * grids.weights
            idx = abs(rw) > ks.small_rho_cutoff / ngrids
            logger.alldebug1(ks, 'Drop grids %d',
                             grids.weights.size - numpy.count_nonzero(idx))
            grids.coords  = numpy.asarray(grids.coords [idx], order='C')
            grids.weights = numpy.asarray(grids.weights[idx], order='C')

    grids.non0tab = grids.make_mask(cell, grids.coords)

    return grids