# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------ Muon Energy Loss ------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# This classes calculate the stopping power for muons in dense media using two
# different approaches, one analytical and one semi-analytical. The stopping
# power is given in MeV*cm**2/g.
# <-dE/dx> = a(E) + b(E)*E
# ------------------------------------------------------------------------------

import numpy as np
from scipy.integrate import quad
from scipy.integrate import simps
from scipy.special import spence
from scipy import interpolate
import scipy.constants as cst
# import sys
# sys.path.append('../git/MCEq/ParticleDataTool')
# import ParticleDataTool as pd
#
# ParticleData = pd.PYTHIAParticleData()

class MuonEnergyLoss():
    def __init__(self):

        # Masses
        # Using ParticleDataTool
        # self.me = ParticleData.mass(11)*1e3 # MeV/c**2
        # self.M = ParticleData.mass(13)*1e3 # MeV/c**2
        # Using scipy.constants
        self.me = cst.physical_constants['electron mass energy equivalent in MeV'][0]
        self.M = cst.physical_constants['muon mass energy equivalent in MeV'][0]

        # Medium related constants
        self.Z_A = 0.4991
        self.A = 14.6615315
        self.Z = self.Z_A*self.A
        self.I = 85.7*1e-6 # eV
        self.a = 0.10914
        self.k = 3.3994
        self.x0 = 1.7418
        self.x1 = 4.2759
        self.Cbar = 10.5961

        # Physical constants
        self.NA = cst.N_A # mol**(-1)
        self.K_A = 4*np.pi*cst.N_A*(cst.physical_constants['classical electron radius'][0]*100)**2*self.me # MeV g**(-1) cm**2
        self.alpha = cst.alpha
        self.re = cst.physical_constants['classical electron radius'][0]*100 # cm

        # Constants needed to calculate some functions
        self.Dn = 1.54*self.A**0.27
        self.B_nucl = 182.7
        # self.B_nucl = 202.4 # Hydrogen
        self.B_elec = 1429
        # self.B_elec = 446 # Hydrogen
        self.m1_2 = 0.54e6 # MeV**2
        self.m2_2 = 1.8e6 # MeV**2

        # Interpolations needed for photonuclear interaction
        self.sophia_data_eps, self.sophia_data_proton, self.sophia_data_neutron = np.loadtxt('sophia_csec.dat', delimiter=',', unpack='True')
        self.f_proton = interpolate.interp1d(self.sophia_data_eps, self.sophia_data_proton, fill_value='extrapolate')
        self.f_neutron = interpolate.interp1d(self.sophia_data_eps, self.sophia_data_neutron, fill_value='extrapolate')

        # Vectorized functions
        self.vec_b_brems = np.vectorize(self.b_brems)
        self.vec_b_pair = np.vectorize(self.b_pair)
        self.vec_b_photo_nucl = np.vectorize(self.b_photo_nucl)

        # Integration limits
        self.nu_low = 1e-15
        self.nu_high = 1-1e-15

    # Electronic contribution
    def qmax(self, E):
        beta = self.beta(E)
        gamma = self.gamma(E)
        return ((2*self.me*beta**2*gamma**2) /
                (1 + 2*gamma*self.me/self.M
                   + (self.me/self.M)**2))

    def p_tot(self, E):
        return (E**2 - self.M**2)**(1./2)

    def beta(self, E):
        return self.p_tot(E)/E

    def gamma(self, E):
        return E/self.M

    def delta_dens(self, E):
        beta = self.beta(E)
        gamma = self.gamma(E)

        x = np.asarray(np.log10(beta*gamma))

        delta_dens = np.zeros(x.size)

        coef_1 = lambda xv: self.a*(self.x1-xv)**self.k

        ind_high = np.where(x >= self.x1)
        ind_med = np.logical_and(self.x1 > x, x >= self.x0)
        ind_low = np. where(x < self.x0)

        delta_dens[ind_high] = 2*np.log(10)*x[ind_high] - self.Cbar
        delta_dens[ind_med] = 2*np.log(10)*x[ind_med] - self.Cbar + coef_1(x[ind_med])
        delta_dens[ind_low] = 0

        return delta_dens

    def Delta_brems_atelec(self, E):
        return (
            self.K_A*(self.Z_A)*self.alpha/(4*np.pi) * (np.log(2*E/self.M) -
                (1./3)*np.log(2*self.qmax(E)/self.me)) * np.log(2*self.qmax(E)/self.me)**2
            )

        return Delta_brems_atelec

    def a_electronic(self, E):
        beta = self.beta(E)
        gamma = self.gamma(E)
        return (self.K_A*(self.Z_A)*(1./beta**2)
            * ((1./2)*np.log(2*self.me*beta**2*gamma**2*self.qmax(E)/self.I**2)
                - beta**2 - self.delta_dens(E)/2 + self.qmax(E)**2/(8*(gamma*self.M)**2))
            + self.Delta_brems_atelec(E))

    # Bremsstrahlung contribution
    def delta_brems(self, nu, E):
        return self.M**2*nu/(2*E*(1-nu))

    def upperdelta_brems(self, nu, E):
        delta = self.delta_brems(nu, E)
        return np.log(self.Dn / (1+delta*(self.Dn*np.sqrt(np.e)-2)/self.M))

    def phi_brems_nucl(self, nu, E):
        delta = self.delta_brems(nu, E)
        upperdelta_brems = self.upperdelta_brems(nu, E)
        return np.log((self.B_nucl*self.M*self.Z**(-1./3)/self.me) / (1+delta*np.sqrt(np.e)*self.B_nucl*self.Z**(-1./3)/self.me)) - upperdelta_brems

    def csec_brems_nucl(self, nu, E):
        phi_brems_nucl = self.phi_brems_nucl(nu, E)
        return self.alpha * (2*self.Z*(self.me/self.M)*self.re)**2 * (4./3-(4./3)*nu+nu**2) * phi_brems_nucl/nu

    def phi_brems_elec(self, nu, E):
        delta = self.delta_brems(nu, E)
        return np.log((self.M/delta) / (self.M*delta/self.me**2 + np.sqrt(np.e))) - np.log(1+self.me/(delta*self.B_elec*self.Z**(-2./3)*np.sqrt(np.e)))

    def csec_brems_elec(self, nu, E):
        phi_brems_elec = self.phi_brems_elec(nu, E)
        return self.alpha*self.Z * (2*(self.me/self.M)*self.re)**2 * (4./3-(4./3)*nu+nu**2) * phi_brems_elec/nu

    def csec_brems(self, nu, E):
        return self.csec_brems_nucl(nu, E) + self.csec_brems_elec(nu, E)

    def b_brems_integrand(self, nu, E):
        return nu*self.csec_brems(nu, E)

    def b_brems(self, E):
        integral = quad(self.b_brems_integrand, self.nu_low, self.nu_high, E)
        result = (self.NA/self.A)*integral[0]
        return result

    # Pair production contribution
    def b_pair(self, E):
        return 0

    # Photonuclear interaction contribution
    def t(self, nu):
        return self.M**2*nu**2/(1-nu)

    def kappa(self, nu):
        return 1 - 2./nu + 2./nu**2

    def x_photo_nucl(self, eps):
        # This function needs sigma to be in microbarn.
        sigma = self.sigma_photo_nucl(eps)
        return 0.00282*self.A**(1./3)*sigma*1e30

    def G_photo_nucl(self, eps):
        x = self.x_photo_nucl(eps)
        with np.errstate(all='ignore'):
            return np.nan_to_num((3./x**3)*(x**2/2 - 1 + np.exp(-x)*(1+x)))

    def sigma_photo_nucl(self, eps):
        # Energy must be given in MeV as an argument, while used as GeV in this
        # function. f_nucleon functions return values expressed in  microbarn.
        # We convert those to cm**2 in this function for consistency with the
        # rest of the code.
        eps_gev = eps*1e-3
        return (self.f_proton(eps_gev)*self.Z_A + self.f_neutron(eps_gev)*(self.A-self.Z)/self.A)*1e-30

    def csec_photo_nucl(self, nu, E):
        eps = nu*E
        sigma = self.sigma_photo_nucl(eps)
        G = self.G_photo_nucl(eps)
        kappa = self.kappa(nu)
        t = self.t(nu)
        return (
            (self.alpha*self.A)/(2*np.pi)*sigma*nu * (0.75*G*(kappa*np.log(1+self.m1_2/t) - kappa*self.m1_2/(self.m1_2+t) - 2*self.M**2/t) +
                0.25*(kappa*np.log(1+self.m2_2/t) - 2*self.M**2/t) +
                self.M**2/(2*t)*(0.75*G*self.m1_2/(self.m1_2+t) + 0.25*(self.m2_2/t)*np.log(1+t/self.m2_2)))
            )

    def b_photo_nucl_integrand(self, nu, E):
        return nu*self.csec_photo_nucl(nu, E)

    def b_photo_nucl(self, E):
        integral = quad(self.b_photo_nucl_integrand, self.nu_low, self.nu_high, E)
        result = (self.NA/self.A)*integral[0]
        return result

    # Total radiative losses
    def radiative_loss(self, E):
        return (self.vec_b_brems(E)+self.vec_b_pair(E)+self.vec_b_photo_nucl(E))*E

    # Stopping power
    def stopping_power(self, E):
        return self.a_electronic(E) + self.radiative_loss(E)

    # Comparison with PDG tables
    def load_pdg(self, filename1, *filename2):
        # Filename must be a string.
        pdg_data = np.loadtxt(filename1, skiprows=59, usecols=[0,2,3,4,5,6,7], unpack=True)
        self.pdg_T = pdg_data[0]
        self.pdg_ionization = pdg_data[1]
        self.pdg_brems = pdg_data[2]
        self.pdg_pair = pdg_data[3]
        self.pdg_photonuc = pdg_data[4]
        self.pdg_radloss = pdg_data[5]
        self.pdg_stopping_power = pdg_data[6]

        self.pdg_E = self.pdg_T + self.M

        pdg_data_b = np.loadtxt(filename2[0], skiprows=6, unpack=True)
        self.pdg_E_b = pdg_data_b[0]*1e3
        self.pdg_b_brems = pdg_data_b[1]
        self.pdg_b_pair = pdg_data_b[2]
        self.pdg_b_nucl = pdg_data_b[3]
        self.pdg_b_total = pdg_data_b[4]

    def comparison_pdg(self, filename1):
        pdg_data = np.loadtxt(filename1, skiprows=59, usecols=[0,2,3,4,5,6,7], unpack=True)
        pdg_T = pdg_data[0]
        pdg_ionization = pdg_data[1]
        pdg_brems = pdg_data[2]
        pdg_pair = pdg_data[3]
        pdg_photonuc = pdg_data[4]
        pdg_radloss = pdg_data[5]
        pdg_stopping_power = pdg_data[6]
        pdg_E = pdg_T + self.M

        ionization = self.a_electronic(pdg_E)
        brems = self.vec_b_brems(pdg_E)*pdg_E
        pair = self.vec_b_pair(pdg_E)*pdg_E
        photonuc = self.vec_b_photo_nucl(pdg_E)*pdg_E
        radloss = brems+pair+photonuc
        stopping_power = ionization + radloss

        delta_ionization = np.absolute(pdg_ionization-ionization)
        delta_brems = np.absolute(pdg_brems-brems)
        delta_pair = np.absolute(pdg_pair-pair)
        delta_photonuc = np.absolute(pdg_photonuc-photonuc)
        delta_radloss = np.absolute(pdg_radloss-radloss)
        delta_stopping_power = np.absolute(pdg_stopping_power-stopping_power)

        percentage_ionization = delta_ionization*100/np.absolute(pdg_ionization)
        percentage_brems = delta_brems*100/np.absolute(pdg_brems)
        percentage_pair = delta_pair*100/np.absolute(pdg_pair)
        percentage_photonuc = delta_photonuc*100/np.absolute(pdg_photonuc)
        percentage_radloss = delta_radloss*100/np.absolute(pdg_radloss)
        percentage_stopping_power = delta_stopping_power*100/np.absolute(pdg_stopping_power)

        np.savetxt('comparison_pdg_absolute.dat', np.c_[
            pdg_T, delta_ionization, delta_brems, delta_pair, delta_photonuc, delta_radloss, delta_stopping_power
            ], delimiter=',', header='T(MeV),Ionization(MeV cm^2/g),Brems(MeV cm^2/g),Pair(MeV cm^2/g),Photonuc(MeV cm^2/g),Radloss(MeV cm^2/g),dE/dx(MeV cm^2/g)')
        np.savetxt('comparison_pdg_relative.dat', np.c_[
            pdg_T, percentage_ionization, percentage_brems, percentage_pair, percentage_photonuc, percentage_radloss, percentage_stopping_power
            ], delimiter=',', header='T(MeV),Ionization(MeV cm^2/g),Brems(MeV cm^2/g),Pair(MeV cm^2/g),Photonuc(MeV cm^2/g),Radloss(MeV cm^2/g),dE/dx(MeV cm^2/g)')

# ------------------------------------------------------------------------------
# ---------------------- Analytical approach by Nikishov. ----------------------
# ------------------------------------------------------------------------------

class MuonEnergyLossNik(MuonEnergyLoss):
    def __init__(self):
        MuonEnergyLoss.__init__(self)

        # Constants needed to calculate some functions
        self.g = 1.95e-5
        # self.g = 4.4e-5 # Hydrogen

    # Pair production contribution
    def csec_pair_nucl_args(self, nu, E):
        self.theta = self.me**2/self.M**2
        self.z = nu**2/(self.theta*(1-nu))
        self.Bz = np.sqrt(1./4+1./self.z)
        self.z1 = self.Bz -1./2
        self.z2 = self.Bz + 1./2
        self.y = (self.z1+self.z2)/self.z2**2
        self.s = E*np.sqrt(1-nu)*self.Z**(1./3)/(self.M*151)
        self.w = self.s*np.sqrt(self.z)
        self.Bw = np.sqrt(1./4+1./self.w)
        self.w1 = self.Bw -1./2
        self.w2 = self.Bw + 1./2
        self.u = self.w + self.z
        self.Bu = np.sqrt(1./4+1./self.u)
        self.u1 = self.Bu -1./2
        self.u2 = self.Bu + 1./2

    def f1_theta_f3(self):
        return (
            44./(45*self.z) - 16./45 - 4.*self.theta/9 - (7./9 + 8.*self.z/45 + 7.*self.z*self.theta/18)*np.log(self.z) +
            (16.*self.z/45 + 38./45 - 44./(45*self.z) + 4./(3*(self.z+4)) + (7.*self.z/9 - 2./9 + 8./(3*(self.z+4)))*self.theta)*self.Bz*np.log(self.z2/self.z1)
            )

    def phi2_theta_phi3(self):
        return (
            (7./36 + 2.*self.z/45 + 7.*self.z*self.theta/72)*(np.log(self.z2/self.z1)**2 + np.pi**2 + 2*np.log(self.z)**2) + (7./18 + 3.*self.z/20 + 7.*self.z*self.theta/36)*np.log(self.z) + 653./270 - 28./(9*self.z) + 2.*self.theta/3 +
            (-3*self.z/10 - 92./45 + 52./(45*self.z) + (2./9 - 7.*self.z/18)*self.theta)*self.Bz*np.log(self.z2/self.z1) + self.Bz*(-8.*self.z/45 - 19./45 - 8./(45*self.z) - (2./9 + 7.*self.z/18)*self.theta)*(spence(1-self.y) + 2*spence(1-1/self.z2) + 3*np.log(self.z2/self.z1)**2/2) +
            (8./self.z + self.z*self.theta) * (self.Bz/(3*(self.z+4))) * (6*spence(1-1/self.z2) - spence(1-self.y) + np.log(self.z2/self.z1)**2/2)
            )

    def Jplus1(self):
        return 2*spence(1-1/self.z2) - spence(1-self.y) + np.log(self.z1)*np.log(self.z2/self.z1)

    def Jplus2(self):
        return (
            spence(1-self.u1/self.z1) - spence(1-self.u2/self.z2) + spence(1-self.z1/(self.z1+self.u2)) - spence(1-self.z2/(self.z2+self.u1)) + np.log(self.u1/self.z1)*np.log(1-self.u1/self.z1) -
            np.log(self.u2/self.z2)*np.log(1-self.u2/self.z2) + np.log(self.z2/self.z1)*np.log(self.u*(self.z1+self.u2))
            )

    def Jplus(self):
        return self.Jplus1() + self.Jplus2()

    def Iplus(self):
        return (
            spence(1-self.u1/self.w1) - spence(1-self.u2/self.w2) - 2*spence(1-self.w1/self.w2) + spence(1-self.w1/(self.w1+self.u2)) -
            spence(1-self.w2/(self.w2+self.u1)) + np.pi**2/3 + np.log(self.w2/self.w1)*np.log((self.w1+self.u2)*self.u/(self.w2*self.z)) +
            np.log(self.u1/self.w1)*np.log(1-self.u1/self.w1) - np.log(self.u2/self.w2)*np.log(1-self.u2/self.w2)
            )

    def H(self):
        return (
            spence(1-self.z/(self.u+4)) - spence(1-(self.z+4)/(self.u+4)) + spence(1-self.z/(self.z+4)) - 2*spence(1-self.u/(self.u+4)) +
            spence(1-4*self.w/(self.u*(self.z+4))) + spence(1-4*self.z/(self.u*(self.w+4))) - spence(1-4/(self.w+4)) + np.pi**2/6 +
            2*np.log(self.z1)*np.log(self.z2) - 4*np.log(self.u1)*np.log(self.u2) - np.log(self.z)**2 + np.log(self.z+4)**2 - np.log(1+4./self.w)*np.log(self.u+4) -
            np.log(4*self.w)*np.log(self.z+4) + np.log(16)*np.log(self.u+4) - np.log(self.u+4)**2 + 2*np.log(self.u)**2 +
            np.log(self.u)*np.log((self.z+4)*(self.w+4)/(4*4*self.w)) - np.log(self.z)*np.log((self.z+4)*self.u/(4*self.w))
            )

    def I_pair(self):
        H = self.H()
        Jplus = self.Jplus()
        Jplus2 = self.Jplus2()
        Iplus = self.Iplus()
        return (
            (7./9 + 8.*self.z/45 + 7.*self.z*self.theta/18)*H - (16.*self.z/45 + 38./45 + 16./(45*self.z) + (7.*self.z/9 + 4./9)*self.theta)*self.Bz*Jplus +
            (-16.*self.z/45 - 14./9 - 8./(9*self.w) + 2.*self.z/(45*self.w) - 4.*self.z/(5*self.w**2) + 2.*self.z/(3*(self.w+4)) - (7.*self.z/9 + 4.*self.z/(9*self.w))*self.theta)*self.Bw*Iplus +
            (32.*self.u/(45*self.w) - 88./(45*self.z) - 16/(45*self.w) + 8.*self.z/(5*self.w**2) + 8.*self.u*self.theta/(9*self.w))*self.Bu*np.log(self.u2/self.u1) +
            (68./45 - 16./(45*self.z) + 8./(3*self.w) - 2*self.z/(3*self.w) - 8*self.theta/9)*self.Bz*np.log(self.z2/self.z1) + 104./(45*self.z) -
            8/(15*self.w) - 62./27 - (8/(9*self.w) + self.z/(45*self.w) + 4*self.z/(5*self.w**2) + 4*self.z*self.theta/(9*self.w))*np.log(self.z) +
            (1 + self.z*self.theta/2)*(1./(3*self.w))*(np.log(self.u2/self.u1)**2 - np.log(self.z2/self.z1)**2) +
            (8./self.z + self.z*self.theta)*(self.Bz/(3*(self.z+4))) * (2*Jplus2 + np.log(self.z2)**2 - np.log(self.z1)**2)
            )

    def csec_pair_nucl(self, nu, E):
        self.csec_pair_nucl_args(nu, E)
        eps = nu*E
        f1_theta_f3 = self.f1_theta_f3()
        phi2_theta_phi3 = self.phi2_theta_phi3()
        I = self.I_pair()
        return ((2*self.alpha*self.re*self.Z)**2 * (1-nu) / (np.pi*nu)) * (f1_theta_f3*np.log(2.*eps/self.me) + phi2_theta_phi3 + I)

    def b_pair_integrand(self, nu, E):
        return nu*self.csec_pair_nucl(nu, E)

    def b_pair_elec(self, E):
        return self.Z_A*(0.073*np.log((2*E/self.M)/(1+self.g*self.Z**(2./3)*E/self.M)) - 0.31)*1e-6

    def b_pair(self, E):
        integral = quad(self.b_pair_integrand, self.nu_low, self.nu_high, E)
        result = (self.NA/self.A)*integral[0] + self.b_pair_elec(E)
        return result

# ------------------------------------------------------------------------------
# ----------------- Semi-analytical approach by Kokoulin et al. ----------------
# ------------------------------------------------------------------------------

class MuonEnergyLossKok(MuonEnergyLoss):
    def __init__(self):
        MuonEnergyLoss.__init__(self)

        # Constants needed to calculate some functions
        self.Astar = 183
        self.gamma1 = 1.95e-5
        self.gamma2 = 5.30e-5
        # self.Astar = 202.4 # Hydrogen
        # self.gamma1 = 4.4e-5 # Hydrogen
        # self.gamma2 = 4.8e-5 # Hydrogen

        # Vectorized functions
        self.vec_G_pair_integral = np.vectorize(self.G_pair_integral)
        self.vec_b_pair_integrand = np.vectorize(self.b_pair_integrand)

        # Integration precision for pair production
        self.quad_precision = 1e-3

    # Pair production contribution
    def zeta(self, E):
        if E <= 35*self.M:
            result = 0
        elif E > 35*self.M:
            num = 0.073*np.log((E/self.M)/(1+self.gamma1*self.Z**(2./3)*E/self.M)) - 0.26
            if num < 0:
                result = 0
            elif num >= 0:
                den = 0.058*np.log((E/self.M)/(1+self.gamma2*self.Z**(1./3)*E/self.M)) - 0.14
                result = num/den

        return result

    def G_pair(self, t, rho, beta, xi,nu, E):
        return self.phi(t, rho, beta, xi, nu, E, 1) + (self.me/self.M)**2*self.phi(t, rho, beta, xi, nu, E, 2)

    def phi(self, t, rho, beta, xi, nu, E, particle):
        if particle == 1:
            calculation = self.B_e(t, rho, beta, xi, nu)*self.Lprime_e(t, rho, beta, xi, nu, E)
        if particle == 2:
            calculation = self.B_mu(t, rho, beta, xi, nu)*self.Lprime_mu(t, rho, beta, xi, nu, E)
        if calculation < 0:
            result = 0
        elif calculation >= 0:
            result = calculation

        return calculation

    def B_e(self, t, rho, beta, xi, nu):
        if xi < 1e3:
            result = ((2+rho**2)*(1+beta) + xi*(3+rho**2))*np.log(1+1/xi) + (1-rho**2-beta)/(1+xi) - (3+rho**2)
        elif xi >= 1e3:
            result = (1/(2*xi)) * ((3-rho**2) + 2*beta*(1+rho**2))

        return result

    def B_mu(self, t, rho, beta, xi, nu):
        if xi <= 1e-3:
            result = (xi/2) * ((5-rho**2) + beta*(3+rho**2))
        elif xi > 1e-3:
            result = ((1+rho**2)*(1+3*beta/2) - (1+2*beta)*(1-rho**2)/xi)*np.log(1+xi) + (xi*(1-rho**2-beta)/(1+xi)) + (1+2*beta)*(1-rho**2)

        return result

    def xi(self, t, rho, nu):
        return self.M**2*nu**2*(1-rho**2)/(4*self.me**2*(1-nu))

    def beta_pair(self, nu):
        return nu**2/2*(1-nu)

    def Lprime_e(self, t, rho, beta, xi, nu, E):
        Y_e = self.Y_e(t, rho, beta, xi, nu)
        num = self.Astar*self.Z**(-1./3)*np.sqrt((1+xi)*(1+Y_e))
        den = 1 + (2*self.me*np.sqrt(np.e)*self.Astar*self.Z**(-1./3)*(1+xi)*(1+Y_e))/(E*nu*(1-rho**2))
        return np.log(num/den) - np.log(1 + (3*self.me*self.Z**(-1./3)/(2*self.M))**2*(1+xi)*(1+Y_e))/2

    def Lprime_mu(self, t, rho, beta, xi, nu, E):
        Y_mu = self.Y_mu(t, rho, beta, nu)
        num = (self.M/self.me)*self.Astar*self.Z**(-1./3)*np.sqrt((1+1/xi)*(1+Y_mu))
        den = 1 + (2*self.me*np.sqrt(np.e)*self.Astar*self.Z**(-1./3)*(1+xi)*(1+Y_mu))/(E*nu*(1-rho**2))
        return np.log(num/den) - np.log((3./2)*self.Z**(1./3)*np.sqrt((1+1/xi)*(1+Y_mu)))

    def Y_e(self, t, rho, beta, xi, nu):
        num = 5 - rho**2 + 4*beta*(1+rho**2)
        den = 2*(1+3*beta)*np.log(3+1/xi) - rho**2 - 2*beta*(2-rho**2)
        return num/den

    def Y_mu(self, t, rho, beta, nu):
        num = 4 + rho**2 + 3*beta*(1+rho**2)
        den = (1+rho**2)*((3./2)+2*beta)*np.log(3+rho) + 1 - (3./2)*rho**2
        return num/den

    def tmin(self, eps, E):
        num = 4*self.me/eps + 12*self.M**2/(E*(E-eps))*(1-4*self.me/eps)
        den = 1 + (1-6*self.M**2/(E*(E-eps)))*np.sqrt(1-4*self.me/eps)
        return np.log(num/den)

    def G_pair_integrand(self, t, nu, E):
        rho = 1 - np.exp(t)
        beta = self.beta_pair(nu)
        xi = self.xi(t, rho, nu)
        return self.G_pair(t, rho, beta, xi, nu, E)*np.exp(t)

    def G_pair_integral(self, t_low, nu, E):
        return quad(self.G_pair_integrand, t_low, 0, (nu, E), epsrel=self.quad_precision)[0]

    def csec_pair(self, nu, E):
        eps = nu*E
        tmin = self.tmin(eps, E)
        zeta = self.zeta(E)
        Gintegral = self.vec_G_pair_integral(tmin, nu, E)
        return (4*self.Z*(self.Z+zeta)*self.NA*(self.alpha*self.re)**2)/(3*np.pi*self.A) * (1-nu)/nu * Gintegral

    def b_pair_integrand(self, nu, E):
        return nu*self.csec_pair(nu, E)

    def b_pair(self, E):
        # To avoid the lower and upper limit of the domain of the function.
        nu_low = (4*self.me/E) + (4*self.me/E)/1000
        nu_high = ((E - (3*np.sqrt(np.e)/4)*self.M*self.Z**(1./3))/E) - ((E - (3*np.sqrt(np.e)/4)*self.M*self.Z**(1./3))/E)/1000
        npoints = 6*np.log10(E) - 1.7
        ev_points = np.logspace(np.log10(nu_low), np.log10(nu_high), npoints)
        ev_vals = self.vec_b_pair_integrand(ev_points, E)
        return simps(ev_vals, ev_points)
