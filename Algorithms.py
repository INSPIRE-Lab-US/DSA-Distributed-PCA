import numpy as np
import math


class Algorithms():
    def __init__(self, data, iterations, K, num_nodes, initial_est, ground_truth):

        self.data = data  # data samples
        self.num_itr = iterations  # Number of iterations
        self.K = K  # dimension of eigenspace to be estimated
        self.n = num_nodes  # Number of nodes n
        self.X_init = initial_est  # initial estimate of K-dimensional eigenspace (dxK)
        self.X_gt = ground_truth  # true K-dimensional eigenspace (dxK)

    def OI(self):
        N = self.data.shape[1]
        Cy = (1 / N) * np.dot(self.data, self.data.transpose())
        X_oi = self.X_init
        angle_oi = self.dist_subspace(self.X_gt, X_oi)
        for i in range(self.num_itr):
            X_oi = np.dot(Cy, X_oi)
            X_oi, r = np.linalg.qr(X_oi)
            angle_oi = np.append(angle_oi, self.dist_subspace(self.X_gt, X_oi))
        return angle_oi

    def SeqPM(self):
        N = self.data.shape[1]
        d = self.data.shape[0]
        Cy = (1 / N) * np.dot(self.data, self.data.transpose())
        Cy1 = Cy
        X_final = self.X_init.copy()
        angle_seqpm = self.dist_subspace1(self.X_gt, self.X_init)
        for k in range(self.K):
            X_distpm = self.X_init[:, k]
            for p in range(int(self.num_itr/self.K)):
                X_distpm = np.dot(Cy1, X_distpm)
                X_distpm = X_distpm / np.linalg.norm(X_distpm)
                X_final[:, k] = X_distpm
                angle_seqpm = np.append(angle_seqpm, self.dist_subspace1(self.X_gt, X_final))
            Cy1 = np.dot(np.identity(d) - np.dot(X_final[:, 0:k+1], X_final[:, 0:k+1].transpose()), Cy)
        return angle_seqpm

    def centralized_sanger(self, alpha, step_flag):
        N = self.data.shape[1]
        Cy = (1 / N) * np.dot(self.data, self.data.transpose())
        X_sanger = self.X_init
        angle_sanger = self.dist_subspace(self.X_gt, X_sanger)
        alpha0 = None
        for itr in range(self.num_itr):
            if step_flag == 0:
                alpha0 = alpha  # constant step-size
            elif step_flag == 1:
                alpha0 = alpha / (itr + 1)**0.2  # decreasing step-size
            elif step_flag == 2:
                alpha0 = alpha/math.sqrt(itr + 1)
            X_sanger = X_sanger - alpha0 * self.sanger_centralized_update(Cy, X_sanger)
            angle_sanger = np.append(angle_sanger, self.dist_subspace(self.X_gt, X_sanger))
        return angle_sanger

    def sanger_centralized_update(self, C, X):
        T = np.dot(np.dot(X.transpose(), C), X)
        T = np.triu(T)
        g = -np.dot(C, X) + np.dot(X, T)
        return g

    def DSA(self, WW, alpha, step_flag):
        angle_dsa = self.dist_subspace(self.X_gt, self.X_init)
        N = self.data.shape[1]
        Cy_cell = np.zeros((self.n,), dtype=np.object)
        s = math.floor(N / self.n)
        for i in range(self.n):  # loop for nodes
            Yi = self.data[:, i * s:(i + 1) * s]
            Cy_cell[i] = (1 / s) * np.dot(Yi, Yi.transpose())
        alpha0 = None
        X_dsa = np.tile(self.X_init.transpose(), (self.n, 1))
        for itr in range(self.num_itr):
            if step_flag == 0:
                alpha0 = alpha
            elif step_flag == 1:
                alpha0 = alpha / (itr + 1)**0.2
            elif step_flag == 2:
                alpha0 = alpha / math.sqrt(itr + 1)
            X_dsa = np.dot(WW, X_dsa) - alpha0 * self.sanger_dist_update(Cy_cell, X_dsa)
            err = 0
            for i in range(self.n):
                X1 = X_dsa[i * self.K:(i + 1) * self.K, :]
                X2 = X1.transpose()
                err = err + self.dist_subspace(self.X_gt, X2)
            angle_dsa = np.append(angle_dsa, err / self.n)
        return angle_dsa


    def sanger_dist_update(self, Cell, X):
        grad = np.zeros(X.shape)
        for i in range(Cell.shape[0]):
            X1 = X[i * self.K:(i + 1) * self.K, :]
            X2 = X1.transpose()
            T = np.dot(np.dot(X1, Cell[i]), X2)
            T = np.triu(T)
            g = -np.dot(Cell[i], X2) + np.dot(X2, T)
            grad[i * self.K:(i + 1) * self.K, :] = g.transpose()
        return grad

    def seqdistPM(self, W, Tc):
        N = self.data.shape[1]
        d = self.data.shape[0]
        Cy_cell = np.zeros((self.n,), dtype=np.object)
        s = math.floor(N / self.n)
        for i in range(self.n):  # loop for nodes
            Yi = self.data[:, i * s:(i + 1) * s]
            Cy_cell[i] = (1 / s) * np.dot(Yi, Yi.transpose())
        X_final = np.tile(self.X_init.transpose(), (self.n, 1))
        angle_distpm = self.dist_subspace(self.X_gt, self.X_init)
        angle_distpm1 = np.tile(angle_distpm, (Tc, 1))
        Tp = math.floor(self.num_itr / (Tc))
        for k in range(self.K):
            for i in range(self.n):
                Cy_cell[i] = np.dot(np.eye(d) - np.dot(X_final[i * self.K:i * self.K + k, :].transpose(), X_final[i * self.K:i * self.K + k, :]), Cy_cell[i])
            X_distpm = np.tile(self.X_init[:, k].transpose(), (self.n, 1))
            for p in range(Tp):
                for i in range(self.n):  # loop for nodes
                    C_yi = Cy_cell[i]
                    X1 = X_distpm[i, :]
                    Xi = np.dot(C_yi, X1.transpose())
                    X_distpm[i, :] = Xi.transpose()
                for c in range(Tc):
                    X_distpm = np.dot(W, X_distpm)
                X_distpm1 = X_distpm.transpose()
                X_distpm1 = X_distpm1 / np.linalg.norm(X_distpm1, axis=0)
                X_distpm = X_distpm1.transpose()
                err = 0
                for i in range(self.n):
                    X_final[i * self.K + k, :] = X_distpm[i, :]
                    X1 = X_final[i * self.K:(i + 1) * self.K, :]
                    X2 = X1.transpose()
                    err = err + self.dist_subspace(self.X_gt, X2)
                angle_distpm = np.append(angle_distpm, err / self.n)
                angle_distpm1 = np.append(angle_distpm1, np.tile(err / self.n, (Tc/self.K, 1)))
        return angle_distpm1

    def distributed_OI(self, W, T_consensus):
        N = self.data.shape[1]
        s = math.floor(N / self.n)
        # print('number of samples on one sites:',s)
        covariance_matrix = np.zeros((self.n,), dtype=np.object)
        for i in range(self.n):  # loop for nodes
            Yi = self.data[:, i * s:(i + 1) * s]
            covariance_matrix[i] = (1 / s) * np.dot(Yi, Yi.transpose())

        error = np.zeros((self.n,), dtype=np.object)
        error_index = [0]
        Q = np.zeros((self.n,), dtype=np.object)

        err_curr = np.zeros((self.n, 1))

        for i in range(self.n):
            Q[i] = self.X_init
            Q[i], R = np.linalg.qr(Q[i])
            err_curr[i] = self.dist_subspace(self.X_gt, Q[i])
            error[i] = err_curr[i]
        Tp = int(self.num_itr / T_consensus)
        for t in range(Tp):
            # if t % 100 == 0:
            #     print(100 * (t / doi_itr), '%')

            for i in range(self.n):
                Q[i] = np.dot(covariance_matrix[i], Q[i])

            for t_c in range(T_consensus):
                prev_Q = Q
                Q = self.w_matmul(W, X=Q)

            # 增加内循环中的error
            for i in range(self.n):
                error[i] = np.append(error[i], [err_curr[i]] * T_consensus)

            for i in range(self.n):
                Q[i], R = np.linalg.qr(Q[i])  # Q[i].shape--dim x r
                err_curr[i] = self.dist_subspace(self.X_gt, Q[i])
                error[i] = np.append(error[i], err_curr[i])
            error_index.append(len(error[0]) - 1)

        err_avg = (1 / self.n) * np.sum(error)

        return err_avg

    def distributed_OI_INC(self, W, T_consensus_max, T_consensus_init=1, Tc_inc=1):
        N = self.data.shape[1]
        s = math.floor(N / self.n)
        covariance_matrix = np.zeros((self.n,), dtype=np.object)
        for i in range(self.n):  # loop for nodes
            Yi = self.data[:, i * s:(i + 1) * s]
            covariance_matrix[i] = (1 / s) * np.dot(Yi, Yi.transpose())

        error = np.zeros((self.n,), dtype=np.object)
        error_index = [0]

        err_curr = np.zeros((self.n, 1))
        Q = np.zeros((self.n,), dtype=np.object)

        T_c = [T_consensus_init]
        T_consensus = T_consensus_init
        for i in range(self.n):
            Q[i] = self.X_init
            Q[i], R = np.linalg.qr(Q[i])
            err_curr[i] = self.dist_subspace(self.X_gt, Q[i])
            error[i] = err_curr[i]
        Tp = int(self.num_itr / T_consensus_max)
        for t in range(Tp):
            for i in range(self.n):
                Q[i] = np.dot(covariance_matrix[i], Q[i])
            if t > 1:
                T_consensus = min(int(T_consensus_init + Tc_inc * t), T_consensus_max)
                T_c.append(T_consensus)

            for t_c in range(T_consensus):
                Q = self.w_matmul(W, X=Q)
            for i in range(self.n):
                error[i] = np.append(error[i], [err_curr[i]] * T_consensus)

            for i in range(self.n):
                Q[i], R = np.linalg.qr(Q[i])  # Q[i].shape--dim x r
                err_curr[i] = self.dist_subspace(self.X_gt, Q[i])
                error[i] = np.append(error[i], err_curr[i])
            error_index.append(len(error[0]) - 1)
        err_avg = (1 / self.n) * np.sum(error)
        return err_avg

    def w_matmul(self, W, X):
        n = W.shape[0]
        X_update = np.zeros((n,), dtype=np.object)
        for j in range(n):
            x_temp = np.zeros((n, X[0].shape[0], X[0].shape[1]))
            for i in range(n):
                x_temp[i, :, :] = W[j, i] * X[i]
            X_update[j] = np.sum(x_temp, axis=0)
        return X_update


    def distProjGD(self, WW, alpha, step_flag):
        X_dpgd = np.tile(self.X_init.transpose(), (self.n, 1))
        angle_dpgd = self.dist_subspace(self.X_gt, self.X_init)
        N = self.data.shape[1]
        Cy_cell = np.zeros((self.n,), dtype=np.object)
        s = math.floor(N / self.n)
        for i in range(self.n):  # loop for nodes
            Yi = self.data[:, i * s:(i + 1) * s]
            Cy_cell[i] = (1 / s) * np.dot(Yi, Yi.transpose())
        alpha0 = None
        for itr in range(self.num_itr):
            if step_flag == 0:
                alpha0 = alpha
            elif step_flag == 1:
                alpha0 = alpha / (itr + 1)**0.2
            elif step_flag == 2:
                alpha0 = alpha / math.sqrt(itr + 1)
            err = 0
            V_dpgd = np.dot(WW, X_dpgd)
            for i in range(self.n):
                Ci = Cy_cell[i]
                V1_dpgd = V_dpgd[i * self.K:(i + 1) * self.K, :].transpose() + \
                          alpha0 * np.matmul(Ci, X_dpgd[i * self.K:(i + 1) * self.K, :].transpose())
                V1_dpgd, r = np.linalg.qr(V1_dpgd)  # projection step
                X_dpgd[i * self.K:(i + 1) * self.K, :] = V1_dpgd.transpose()
                err = err + self.dist_subspace(self.X_gt, V1_dpgd)
            angle_dpgd = np.append(angle_dpgd, err / self.n)
        return angle_dpgd


    def DeEPCA(self, WW, K_fastmix):
        N = self.data.shape[1]
        S = np.tile(self.X_init.transpose(), (self.n, 1))
        X = np.tile(self.X_init.transpose(), (self.n, 1))
        X_prev = np.tile(self.X_init.transpose(), (self.n, 1))
        angle_deepca = self.dist_subspace(self.X_gt, self.X_init)
        angle_deepca1 = np.tile(angle_deepca, (K_fastmix, 1))
        Cy_cell = np.zeros((self.n,), dtype=np.object)
        s = math.floor(N / self.n)
        for i in range(self.n):  # loop for nodes
            Yi = self.data[:, i * s:(i + 1) * s]
            Cy_cell[i] = (1 / s) * np.dot(Yi, Yi.transpose())
        for i in range(self.n):
            S1 = S[i * self.K:(i + 1) * self.K, :].transpose() + np.dot(Cy_cell[i], self.X_init) - self.X_init
            S[i * self.K:(i + 1) * self.K, :] = S1.transpose()
        S = self.FastMix(S, K_fastmix, WW, self.K)
        err = 0
        for i in range(self.n):
            S2 = S[i * self.K:(i + 1) * self.K, :].transpose()
            X1, r = np.linalg.qr(S2)
            X1 = self.SignAdjust(X1, self.X_init)
            X_prev[i * self.K:(i + 1) * self.K, :] = X[i * self.K:(i + 1) * self.K, :]
            X[i * self.K:(i + 1) * self.K, :] = X1.transpose()
            err = err + self.dist_subspace(self.X_gt, X1)
        angle_deepca1 = np.append(angle_deepca1, np.tile(err / self.n, (K_fastmix, 1)))
        Tp = int(self.num_itr / K_fastmix)
        for itr in range(Tp):
            for i in range(self.n):
                Xx = X[i * self.K:(i + 1) * self.K, :].transpose()
                Xx_prev = X_prev[i * self.K:(i + 1) * self.K, :].transpose()
                S1 = S[i * self.K:(i + 1) * self.K, :].transpose() + np.dot(Cy_cell[i], Xx) - np.dot(Cy_cell[i], Xx_prev)
                S[i * self.K:(i + 1) * self.K, :] = S1.transpose()
            S = self.FastMix(S, K_fastmix, WW, self.K)
            err = 0
            for i in range(self.n):
                S2 = S[i * self.K:(i + 1) * self.K, :].transpose()
                X1, r = np.linalg.qr(S2)
                X1 = self.SignAdjust(X1, self.X_init)
                X_prev[i * self.K:(i + 1) * self.K, :] = X[i * self.K:(i + 1) * self.K, :]
                X[i * self.K:(i + 1) * self.K, :] = X1.transpose()
                err = err + self.dist_subspace(self.X_gt, X1)
            angle_deepca1 = np.append(angle_deepca1, np.tile(err / self.n, (K_fastmix, 1)))
        return angle_deepca1

    def SignAdjust(self, X, X0):
        k = X0.shape[1]
        for i in range(k):
            if np.dot(X[:, i].transpose(), X0[:, i]) < 0:
                X[:, i] = -X[:, i]
        return X

    def FastMix(self, S, K, WW, dim):
        S_prev = S
        eig_w = np.linalg.eigvalsh(WW)
        eig_w1 = np.unique(eig_w)
        eta_w = (1 - np.sqrt(1 - (eig_w1[-dim - 1]) ** 2)) / (1 + np.sqrt(1 - (eig_w1[-dim - 1]) ** 2))
        for i in range(K):
            S1 = (1 + eta_w) * np.dot(WW, S) - eta_w * S_prev
            S_prev = S
            S = S1
        return S


    def dist_subspace(self, X, Y):
        X = X/np.linalg.norm(X, axis=0)
        Y = Y/np.linalg.norm(Y, axis=0)
        M = np.matmul(X.transpose(), Y)
        sine_angle = 1 - np.diag(M)**2
        dist = np.sum(sine_angle)/X.shape[1]
        return dist
