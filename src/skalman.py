import numpy as np

class Skalman:
    """ Naive implementation of Kalman filter. 
        https://www.kalmanfilter.net/multiSummary.html
    """
    def __init__(
        self,
        n_dim_obs,
        n_dim_state,
        initial_state_mean,
        initial_state_covariance,
        transition_matrices,
        observation_matrices,
        observation_covariance,
        transition_covariance
        ):
        self.n_dim_obs = n_dim_obs
        self.n_dim_state = n_dim_state
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance

        self.F = np.atleast_2d( transition_matrices)
        self.H = np.atleast_2d( observation_matrices)
        self.Q = np.atleast_2d( transition_covariance)
        self.R = np.atleast_2d( observation_covariance)        
    
    @staticmethod
    def Gu(n=None):
        return 0.0

    def predict(self, filt_state, filt_state_cov):
        # ==== PREDICT =====
        # Transition/Predict State
        # n = 0
        pred_state = self.F @ filt_state + Skalman.Gu()
        # Uncertainty
        pred_state_cov = self.F @ filt_state_cov @ self.F.T + self.Q
        return pred_state, pred_state_cov

    def update(self, pred_state, pred_state_cov, obs, obs_matrix):
        # ===== 2. UPDATE CORRECT =====
        # 1. Kalman Gain.
        K = pred_state_cov @ obs_matrix.T @ np.linalg.inv(obs_matrix @ pred_state_cov @ obs_matrix.T + self.R)
        # 2. Update State.
        filt_state = pred_state + np.squeeze( K @ (obs - obs_matrix @ pred_state) )
        # 3. Update uncertainty.
        I = np.eye(pred_state_cov.shape[0])
        filt_state_cov = (I-K @ obs_matrix) @ pred_state_cov @ (I-K @ obs_matrix).T + K @ self.R @ K.T
        return filt_state, filt_state_cov

    def filter(self, z, verbose = False):
        measure = iter(z)
        T = len(z)
        # (time, 0:predicted 1:corrected, values)
        P = np.zeros((T, 2, self.n_dim_state, self.n_dim_state))
        x = np.zeros((T, 2, self.n_dim_state))

        # 1. To align with pykalman, the initial state & cov, is for the predicted state for which there is an observation.
        # That is, initial state and cov is state&cov(0|-1), such that state&cov(0|-1) +  observ(0) => state&cov(0|0)
        # 2. An alternative would be to have initial state & cov to be the state at start (0|0):
        #   state&cov(0|0) => state&cov(1|0) + observ(1) => state&cov(1|1).
        x[0,0] = self.initial_state_mean
        P[0,0] = self.initial_state_covariance
        I = np.eye(self.n_dim_state) # (Identity matrix)
        for n in range(T):
            if n != 0:
                # ==== PREDICT n|n-1 =====
                x[n,0], P[n,0] = self.predict(filt_state=x[n-1,1], filt_state_cov = P[n-1,1])
                print('state({}|{}) = {}'.format(n, n-1, x[n,0])) if verbose else None

            # ===== 1. MEASURE =====
            observation = next(measure)

            # ===== 2. UPDATE CORRECT =====
            x[n,1], P[n,1] = self.update(pred_state=x[n,0], pred_state_cov=P[n,0], obs=observation, obs_matrix=self.H[n])
            print('state({}|{}) = {}'.format(n, n, x[n,1])) if verbose else None

        # Predicted: x[:,0], P[,0]
        # Filtered: x[:,1], P[,1]
        # Skip returning the predicted states, only the filtered states
        return x[:,1], P[:,1]