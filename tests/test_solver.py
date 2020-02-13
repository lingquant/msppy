from msppy.utils.examples import (construct_nvid, construct_nvic,
    construct_nvida, construct_nvidi, construct_nvidinf,
    construct_nvmc, construct_nvm, construct_nvidinfi, construct_nvici)
from msppy.solver import (SDDP, Extensive, SDDiP, Extensive_rolling,
SDDiP_infinity, SDDP_infinity)

class TestSDDP(object):


    def initialize(self):
        self.nvid = construct_nvid()
        self.nvida = construct_nvida()
        self.nvmc = construct_nvmc()

    def test_stopping1(self):
        self.initialize()
        SDDP(self.nvid).solve(
            max_iterations=10
        )
        SDDP(self.nvmc).solve(
            n_processes=3,
            n_steps=3,
            max_iterations=10
        )

    def test_stopping2(self):
        self.initialize()
        SDDP(self.nvid).solve(
            freq_evaluations=1,
            n_simulations=-1,
            tol=1e-4
        )
        SDDP(self.nvmc).solve(
            freq_evaluations=3,
            n_simulations=1000,
            n_processes=3,
            tol=1e-1
        )

    def test_stopping3(self):
        self.initialize()
        SDDP(self.nvid).solve(
            freq_comparisons=1,
            n_simulations=-1,
            tol=1e-4
        )
        SDDP(self.nvmc).solve(
            freq_comparisons=3,
            n_simulations=1000,
            n_processes=3,
            tol=1e-1
        )

    def test_regularize_L1(self):
        self.initialize()
        SDDP(self.nvid).solve(
            max_iterations=10,
            rgl_a=1,
            rgl_b=0.95,
            rgl_norm='L1',
        )

    def test_regularize_L2(self):
        self.initialize()
        SDDP(self.nvid).solve(
            max_iterations=10,
            rgl_a=1,
            rgl_b=0.99,
            rgl_norm='L2',
        )

class TestSDDiP(object):


    def initialize(self):
        self.nvidi = construct_nvidi()

    def test_B_cut(self):
        self.initialize()
        SDDiP(self.nvidi).solve(
            cuts=['B'],
            max_iterations=10,
        )

    def test_SB_cut(self):
        self.initialize()
        SDDiP(self.nvidi).solve(
            cuts=['B','SB'],
            pattern={'in':[0, 3]},
            max_iterations=10,
        )

    def test_LG_cut(self):
        self.initialize()
        SDDiP(self.nvidi).solve(
            cuts=['B','LG'],
            pattern={'cycle':[1, 2]},
            max_iterations=10,
        )

    def test_LG_bin_cut(self):
        self.initialize()
        self.nvidi.binarize()
        SDDiP(self.nvidi).solve(
            cuts=['LG'],
            max_iterations=10,
        )

    def test_continuous(self):
        self.nvici = construct_nvici()
        self.nvici.discretize(n_samples=10)
        SDDiP(self.nvici).solve(
            cuts=['B'],
            max_iterations=10,
        )


class TestSDDPInfinity(object):


    def test_SDDP(self):
        self.nvidinf = construct_nvidinf()
        SDDP_infinity(self.nvidinf).solve(max_iterations=10)

    def test_SDDP_trial_solution_selection(self):
        self.nvidinf = construct_nvidinf()
        SDDP_infinity(self.nvidinf).solve(
            forward_T=7, max_iterations=10)
        self.nvidinf = construct_nvidinf()
        SDDP_infinity(self.nvidinf).solve(
            forward_T=6, max_iterations=10)            

    def test_SDDiP(self):
        self.nvidinfi = construct_nvidinfi()
        SDDiP_infinity(self.nvidinfi).solve(
            cuts=['SB'],
            max_iterations=10
        )



class TestExtensive(object):


    def initialize(self):
        self.nvid = construct_nvid()
        self.nvida = construct_nvida()
        self.nvmc = construct_nvmc()

    def test_extensive(self):
        self.initialize()
        Extensive(self.nvid).solve()
        Extensive(self.nvmc).solve()

    def test_extensive_rolling(self):
        pass
