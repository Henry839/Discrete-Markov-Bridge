def get_sigma(self, t):
    '''
    sigma(t)
    '''
    sigma = self.scheduler(t)

    sigma = sigma.to(self.device)
    sigma = sigma.view(-1,1)

    return sigma



def get_integral(self, t):
    '''
    int_0^t sigma(s) ds
    '''
    sigma = self.integral(t)

    sigma = sigma.to(self.device)
    sigma = sigma.view(-1,1)

    return sigma
