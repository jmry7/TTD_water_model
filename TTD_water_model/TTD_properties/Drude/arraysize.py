import torch

def arraysize(na):
    """
    na:number of atoms
    np:number of particles
    nst:stretch 20-H,1H-H(The H-H stretching is omitted in practical calculations.)
    nb:HOH bend
    ncc:number of core-core intermolecular Coulomb interactions
    ncd:number of core-Drude intermolecular Coulomb internations
    ndd:number of Drude-Drude intermolecular Coulomb interactions
    nbk:number of buckingham interactions
    """
    if na % 3 != 0:
        raise ValueError("The number of atoms (na) must be a multiple of 3 (corresponding to water molecules O, H, H).")
    np=na*2
    nw=na//3
    nst=nw*3
    nb=nw
    nd=nw*3
    ncc=0
    ncd=nw*6
    ndd=nw*3
    nbk=0
    for iw in range(1,nw):
        for jw in range(iw+1,nw+1):
            ncc+=9
            ncd+=18
            ndd+=9
            nbk+=9
    return np, nst, nb, nd, ncc, ncd, ndd, nbk
