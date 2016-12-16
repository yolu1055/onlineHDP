import sys, re, time, string
import numpy as n
from scipy.special import gammaln, psi
import parse_document
from scipy import linalg as LA

n.random.seed(100000001)
convergethresh = 0.0001

def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(n.sum(alpha)))
    return(psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])

def expectation_stick_weights(a_b):
    dig_sum = psi(n.sum(a_b, 0))
    ElogW = psi(a_b[0]) - dig_sum
    Elog1_W = psi(a_b[1]) - dig_sum
    #ElogW[-1] = 0.0
    #Elog1_W[-1] = 0.0
    Elogsticks = n.zeros(len(a_b[0]) + 1)
    Elogsticks[0:-1] = ElogW
    Elogsticks[1:] = ElogW + n.cumsum(Elog1_W)
    #Elogsticks[0] = ElogW[0]
    return Elogsticks 

def E_dirichlet_stickweights(a_b):
    
    dig_sum = psi(n.sum(a_b, 0))
    ElogW = psi(a_b[0]) - dig_sum
    Elog1_W = psi(a_b[1]) - dig_sum
    #ElogW[-1] = 0.0
    #Elog1_W[-1] = 0.0
    
    return (ElogW, Elog1_W)
    

    


class onlineHDP():
    def __init__(self, K, T, D, V, alpha, omega, eta, tau0, kappa):
        
        
        self._K = K
        self._T = T
        self._D = D
        self._V = V
        self._alpha = alpha
        self._omega = omega
        self._eta = eta
        self._tau0 = tau0
        self._kappa = kappa
        self._lambda = 1*n.random.gamma(100., 1./100., (self._K, self._V))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
        self._a_b = n.ones((2, (self._K - 1)))
        self._a_b[1] = self._a_b[1] * self._omega
        self._Elogsticks_1st = expectation_stick_weights(self._a_b)
        self._updatect = 0
        self._rhot = pow(self._tau0 + self._updatect, -self._kappa)
        self._B1 = n.ones(self._lambda.shape)
        self._B_a = n.ones(len(self._a_b[0]))
        self._B_b = n.ones(len(self._a_b[1]))
    
    def calculate_local_likelihood(self, ksi, nphi, phi, u_v, Elogsticks_2nd, Elogbetad):
        Elogc = n.sum(ksi * (self._Elogsticks_1st - n.log(ksi)))
                
        Elogz = n.sum(nphi * (Elogsticks_2nd - n.log(phi)))
                
        (ElogW, Elog1_W) = E_dirichlet_stickweights(u_v)
                
        Elogpai_quote = n.sum(gammaln(u_v[0]) + gammaln(u_v[1]) -\
                               gammaln(n.sum(u_v, axis=0)) - \
                               (u_v[0] - 1) * ElogW +\
                                (self._alpha - u_v[1]) * Elog1_W)
        ElogPw = n.sum(n.dot(Elogbetad, nphi) * ksi.T)
        
        return (ElogPw + Elogpai_quote + Elogc + Elogz)        
        
    def e_step(self, wordids, wordcts):
        batchD = len(wordids)
        likelihood = 0.0
        lambda_hat = n.zeros((self._K, self._V))
        a_hat = n.zeros(self._K - 1)
        b_hat = n.zeros(self._K - 1)
        for d in range(0, batchD):
            likelihood_d = 0.0
            old_likelilood_d = -1e100
            converge = 1.0
            ids = wordids[d]
            cts = wordcts[d]
            Elogbetad = self._Elogbeta[:,ids]
            expElogbetad = self._expElogbeta[:, ids]
            #Elogpai = expectation_stick_weights(self._a_b)
            
            ksi = n.exp(n.ones((self._T, self._K)) * \
            n.sum(Elogbetad, axis = 1))
            ksinorm = n.sum(ksi, axis = 1) + 1e-100
            ksi = ksi / ksinorm[:, n.newaxis]
            
            
            
            phi = n.exp(n.dot(ksi,Elogbetad).T)
            phinorm = n.sum(phi, axis = 1) + 1e-100
            phi = phi / phinorm[:,n.newaxis]
            
            nphi = (phi.T * cts).T
            u_v = n.ones((2, (self._T - 1)))
           
            
            for it in range(0, 100):
                Elogsticks_2nd = expectation_stick_weights(u_v)
                if (it < 3):
                    ksi = n.exp(n.dot(Elogbetad, nphi).T)
                    ksinorm = n.sum(ksi, axis = 1) + 1e-100
                    ksi = ksi / ksinorm[:, n.newaxis]
                    phi = n.exp(n.dot(ksi, Elogbetad).T)
                    phinorm = n.sum(phi, axis = 1) + 1e-100
                    phi = phi / phinorm[:, n.newaxis]
                    nphi = (phi.T * cts).T
                    
                if (it >= 3):
                    ksi = n.exp(self._Elogsticks_1st + \
                            n.dot(Elogbetad, nphi).T)
                    ksinorm = n.sum(ksi, axis = 1) + 1e-100
                    ksi = ksi / ksinorm[:, n.newaxis]
                    phi = n.exp(Elogsticks_2nd + \
                            n.dot(ksi, Elogbetad).T)
                    phinorm = n.sum(phi, axis = 1) + 1e-100
                    phi = phi / phinorm[:, n.newaxis]
                    nphi = (phi.T * cts).T
                
                
                phi_sum = n.sum(nphi, axis = 0)
                u_v[0] = 1 + phi_sum[0:-1]
                reverse_phi_sum = n.flipud(phi_sum[1:])
                phi_cum = n.cumsum(reverse_phi_sum)
                u_v[1] = self._alpha + n.flipud(phi_cum)
                
                
                likelihood_d = self.calculate_local_likelihood(ksi, nphi, phi, u_v,Elogsticks_2nd, Elogbetad)
                converge = abs((likelihood_d - old_likelilood_d) / old_likelilood_d)
                old_likelilood_d = likelihood_d
                likelihood_d = 0.0
                if (converge <= convergethresh):
                    likelihood += likelihood_d
                    break
            
            lambda_hat[:, ids] += n.dot(nphi, ksi).T
            ksi_sum = n.sum(ksi, axis = 0)
            a_hat += ksi_sum[0:-1]
            reverse_ksi_sum = n.flipud(ksi_sum[1:])
            ksi_cum = n.cumsum(reverse_ksi_sum)
            b_hat += n.flipud(ksi_cum)
        
        ss = self._D / batchD
        lambda_hat = self._eta + ss * lambda_hat
        a_hat = 1 + ss * a_hat
        b_hat = self._omega + ss * b_hat
        likelihood = likelihood * ss
        return (likelihood, lambda_hat, a_hat, b_hat)
    
    
    def Adagrad_m_step(self, docs):
        (wordids, wordcts) = parse_document.parse_doc_list(docs, self._V)
        (likelihood, lambda_hat, a_hat, b_hat) = self.e_step(wordids, wordcts)
        g_lambda = -self._lambda + lambda_hat
        g_a = -self._a_b[0] + a_hat
        g_b = -self._a_b[1] + b_hat
        self._B1 = self._B1 + pow(g_lambda, 2)
        self._B_a = self._B_a + pow(g_a, 2)
        self._B_b = self._B_b + pow(g_b, 2)
        #self._rhot = pow(self._tau0 + self._updatect, -self._kappa)
        
        lr_lambda = self._kappa / (1.0 + pow(self._B1, 0.5))
        lr_a = self._kappa /(1.0 + pow(self._B_a, 0.5))
        lr_b = self._kappa /(1.0 + pow(self._B_b, 0.5))
        self._lambda = self._lambda + lr_lambda * g_lambda
        self._a_b[0] = self._a_b[0] + lr_a * g_a
        self._a_b[1] = self._a_b[1] + lr_b * g_b
        
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
        self._Elogsticks_1st = expectation_stick_weights(self._a_b)
        likelihood = likelihood + self.calculate_global_likelihood()
        self.saveparameters(lambda_hat, a_hat, b_hat)
        
        self._updatect = self._updatect + 1
        
        return likelihood
    
    
       
    def m_step(self, docs):
        (wordids, wordcts) = parse_document.parse_doc_list(docs, self._V)
        (likelihood, lambda_hat, a_hat, b_hat) = self.e_step(wordids, wordcts)
        self._rhot = pow(self._tau0 + self._updatect, -self._kappa)
        self._lambda = (1 - self._rhot) * self._lambda + self._rhot * lambda_hat
        self._a_b[0] = (1 - self._rhot) * self._a_b[0] + self._rhot * a_hat
        self._a_b[1] = (1 - self._rhot) * self._a_b[1] + self._rhot * b_hat
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
        self._Elogsticks_1st = expectation_stick_weights(self._a_b)
        likelihood = likelihood + self.calculate_global_likelihood()
        self.saveparameters(lambda_hat, a_hat, b_hat)
        
        self._updatect = self._updatect + 1
        
        return likelihood
        
    def calculate_global_likelihood(self,): 
        likelihood = 0.0
        likelihood = likelihood + n.sum((self._eta-self._lambda) *\
                                         self._Elogbeta)
        likelihood = likelihood + n.sum(gammaln(self._lambda) - \
                                        gammaln(self._eta))
        likelihood = likelihood + n.sum(gammaln(self._eta*self._V) - \
                                        gammaln(n.sum(self._lambda, 1)))
        
        (ElogW, Elog1_W) = E_dirichlet_stickweights(self._a_b)
                
        Elogtheta_quote = n.sum(gammaln(self._a_b[0]) + gammaln(self._a_b[1]) -\
                               gammaln(n.sum(self._a_b, axis=0)) - \
                               (self._a_b[0] - 1) * ElogW +\
                                (self._omega - self._a_b[1]) * Elog1_W +\
                                 gammaln(1 + self._omega) - gammaln(self._omega))   
        
        likelihood = likelihood + Elogtheta_quote
        
        return likelihood
        
    
