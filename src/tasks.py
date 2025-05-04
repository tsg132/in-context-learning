import math

import torch


def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()


def cross_entropy(ys_pred, ys):
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)


class Task:
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(
    task_name, n_dims, batch_size, pool_dict=None, num_tasks=None, **kwargs
):
    task_names_to_classes = {
        "linear_regression": LinearRegression,
        "sparse_linear_regression": SparseLinearRegression,
        "linear_classification": LinearClassification,
        "noisy_linear_regression": NoisyLinearRegression,
        "quadratic_regression": QuadraticRegression,
        "relu_2nn_regression": Relu2nnRegression,
        "decision_tree": DecisionTree,
        
        "rff_regression": RFFRegression,
        "sinusoidal_regression": SinusoidalRegression,
        "rff_fixed": RFFRegressionFixed,
        "sinusoidal_regression_1d": SinusoidalRegression,
        "sinusoidal_regression_10d": SinusoidalRegression,
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError


class RFFRegressionFixed(Task):

    """
    RFF regression with a single fixed kernel sampled at initialization. At each call 
    to evaluate, we sample a new weight vector per example.
    """

    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1.0, rff_dim=16):

        super(RFFRegressionFixed, self).__init__(n_dims, batch_size, pool_dict, seeds)

        self.scale = scale

        self.rff_dim = rff_dim

        self.seeds = seeds

        self.batch_size = batch_size

        if pool_dict is not None:

            self.w_rff = pool_dict['w_rff'].clone()

            self.b_rff = pool_dict['b_rff'].clone()

        else:
            if seeds is not None:

                gen_k = torch.Generator().manual_seed(seeds[0] if isinstance(seeds, (list, tuple)) else seeds)

                self.w_rff = torch.randn(self.rff_dim, self.n_dims, generator=gen_k)

                self.b_rff = 2 * math.pi * torch.rand(self.rff_dim, generator=gen_k)

            else:

                self.w_rff = torch.randn(self.rff_dim, self.n_dims)

                self.b_rff = 2 * math.pi * torch.rand(self.rff_dim)

    def evaluate(self, xs_b):
        device = xs_b.device

        w_rff = self.w_rff.to(device)
        b_rff = self.b_rff.to(device)

        phi_x = torch.cos(torch.einsum('bnd, rd->bnr', xs_b, w_rff) + b_rff)

        phi_x = math.sqrt(2 / self.rff_dim) * phi_x

        b_size = self.batch_size

        if self.seeds is not None:

            w_b = torch.zeros(b_size, self.rff_dim, 1, device=device)

            for i, seed in enumerate(self.seeds):

                gen_i = torch.Generator().manual_seed(seed)

                w_b[i] = torch.randn(self.rff_dim, 1, generator=gen_i, device=device)

        else:

            w_b = torch.randn(b_size, self.rff_dim, 1, device=device)

        ys_b = self.scale * (phi_x @ w_b).squeeze(-1)

        return ys_b
        
    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, rff_dim=32, **kwargs):

        return {"w_rff": torch.randn(rff_dim, n_dims), "b_rff": 2 * math.pi * torch.rand(rff_dim)}
    
    @staticmethod
    def get_metric():

        return squared_error
    
    @staticmethod
    def get_training_metric():

        return mean_squared_error
        
        



    
class RFFRegression(Task):
    
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, rff_dim=32):
        super(RFFRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)

        self.scale = scale
        
        self.rff_dim = rff_dim

        if pool_dict is None and seeds is None:

            self.w_b = torch.randn(self.b_size, self.rff_dim, 1)

            self.w_rff = torch.randn(self.b_size, self.rff_dim, self.n_dims)

            self.b_rff = 2 * math.pi * torch.rand(self.b_size, self.rff_dim, 1)
        
        elif seeds is not None:

            self.w_b = torch.zeros(self.b_size, self.rff_dim, 1)

            self.w_rff = torch.zeros(self.b_size, self.rff_dim, self.n_dims)

            self.b_rff = torch.zeros(self.b_size, self.rff_dim, 1)
            
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.rff_dim, 1, generator=generator)
                
                self.w_rff[i] = torch.randn(self.rff_dim, self.n_dims, generator=generator)

                self.b_rff[i] = 2 * math.pi * torch.rand(self.rff_dim, 1, generator=generator)

        else:
            assert "w_b" in pool_dict and "w_rff" in pool_dict and "b_rff" in pool_dict
            indices = torch.randperm(len(pool_dict["w_b"]))[:batch_size]
            self.w_b = pool_dict["w_b"][indices]
            self.w_rff = pool_dict["w_rff"][indices]
            self.b_rff = pool_dict["b_rff"][indices]

    def evaluate(self, xs_b):

        w_b = self.w_b.to(xs_b.device)
        w_rff = self.w_rff.to(xs_b.device)
        b_rff = self.b_rff.to(xs_b.device)

        phi_x = torch.cos(torch.einsum('bnd, brd->bnr', xs_b, w_rff) + b_rff.transpose(1, 2))
        phi_x = torch.sqrt(torch.tensor(2 / self.rff_dim)) * phi_x

        ys_b = self.scale * (phi_x @ w_b)[:, :, 0]

        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, rff_dim, **kwargs):

        return {
            "w_b": torch.randn(num_tasks, rff_dim, 1),
            "w_rff": torch.randn(num_tasks, rff_dim, n_dims),
            "b_rff": 2 * math.pi * torch.rand(num_tasks, rff_dim, 1),
        }

    @staticmethod
    def get_metric():
        
        return squared_error
    
    @staticmethod
    def get_training_metric():
        return mean_squared_error


class LinearRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.w_b = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class SparseLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        sparsity=3,
        valid_coords=None,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(SparseLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.sparsity = sparsity
        if valid_coords is None:
            valid_coords = n_dims
        assert valid_coords <= n_dims

        for i, w in enumerate(self.w_b):
            mask = torch.ones(n_dims).bool()
            if seeds is None:
                perm = torch.randperm(valid_coords)
            else:
                generator = torch.Generator()
                generator.manual_seed(seeds[i])
                perm = torch.randperm(valid_coords, generator=generator)
            mask[perm[:sparsity]] = False
            w[mask] = 0

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class LinearClassification(LinearRegression):
    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy


class NoisyLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        noise_std=0,
        renormalize_ys=False,
    ):
        """noise_std: standard deviation of noise added to the prediction."""
        super(NoisyLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.noise_std = noise_std
        self.renormalize_ys = renormalize_ys

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        ys_b_noisy = ys_b + torch.randn_like(ys_b) * self.noise_std
        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()

        return ys_b_noisy


class QuadraticRegression(LinearRegression):
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b_quad = ((xs_b**2) @ w_b)[:, :, 0]
        #         ys_b_quad = ys_b_quad * math.sqrt(self.n_dims) / ys_b_quad.std()
        # Renormalize to Linear Regression Scale
        ys_b_quad = ys_b_quad / math.sqrt(3)
        ys_b_quad = self.scale * ys_b_quad
        return ys_b_quad


class Relu2nnRegression(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        hidden_layer_size=100,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(Relu2nnRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.hidden_layer_size = hidden_layer_size

        if pool_dict is None and seeds is None:
            self.W1 = torch.randn(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.randn(self.b_size, hidden_layer_size, 1)
        elif seeds is not None:
            self.W1 = torch.zeros(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.zeros(self.b_size, hidden_layer_size, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.W1[i] = torch.randn(
                    self.n_dims, hidden_layer_size, generator=generator
                )
                self.W2[i] = torch.randn(hidden_layer_size, 1, generator=generator)
        else:
            assert "W1" in pool_dict and "W2" in pool_dict
            assert len(pool_dict["W1"]) == len(pool_dict["W2"])
            indices = torch.randperm(len(pool_dict["W1"]))[:batch_size]
            self.W1 = pool_dict["W1"][indices]
            self.W2 = pool_dict["W2"][indices]

    def evaluate(self, xs_b):
        W1 = self.W1.to(xs_b.device)
        W2 = self.W2.to(xs_b.device)
        # Renormalize to Linear Regression Scale
        ys_b_nn = (torch.nn.functional.relu(xs_b @ W1) @ W2)[:, :, 0]
        ys_b_nn = ys_b_nn * math.sqrt(2 / self.hidden_layer_size)
        ys_b_nn = self.scale * ys_b_nn
        #         ys_b_nn = ys_b_nn * math.sqrt(self.n_dims) / ys_b_nn.std()
        return ys_b_nn

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        return {
            "W1": torch.randn(num_tasks, n_dims, hidden_layer_size),
            "W2": torch.randn(num_tasks, hidden_layer_size, 1),
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class DecisionTree(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, depth=4):

        super(DecisionTree, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.depth = depth

        if pool_dict is None:

            # We represent the tree using an array (tensor). Root node is at index 0, its 2 children at index 1 and 2...
            # dt_tensor stores the coordinate used at each node of the decision tree.
            # Only indices corresponding to non-leaf nodes are relevant
            self.dt_tensor = torch.randint(
                low=0, high=n_dims, size=(batch_size, 2 ** (depth + 1) - 1)
            )

            # Target value at the leaf nodes.
            # Only indices corresponding to leaf nodes are relevant.
            self.target_tensor = torch.randn(self.dt_tensor.shape)
        elif seeds is not None:
            self.dt_tensor = torch.zeros(batch_size, 2 ** (depth + 1) - 1)
            self.target_tensor = torch.zeros_like(dt_tensor)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.dt_tensor[i] = torch.randint(
                    low=0,
                    high=n_dims - 1,
                    size=2 ** (depth + 1) - 1,
                    generator=generator,
                )
                self.target_tensor[i] = torch.randn(
                    self.dt_tensor[i].shape, generator=generator
                )
        else:
            raise NotImplementedError

    def evaluate(self, xs_b):
        dt_tensor = self.dt_tensor.to(xs_b.device)
        target_tensor = self.target_tensor.to(xs_b.device)
        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i in range(xs_b.shape[0]):
            xs_bool = xs_b[i] > 0
            # If a single decision tree present, use it for all the xs in the batch.
            if self.b_size == 1:
                dt = dt_tensor[0]
                target = target_tensor[0]
            else:
                dt = dt_tensor[i]
                target = target_tensor[i]

            cur_nodes = torch.zeros(xs_b.shape[1], device=xs_b.device).long()
            for j in range(self.depth):
                cur_coords = dt[cur_nodes]
                cur_decisions = xs_bool[torch.arange(xs_bool.shape[0]), cur_coords]
                cur_nodes = 2 * cur_nodes + 1 + cur_decisions

            ys_b[i] = target[cur_nodes]

        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class SinusoidalRegression(Task):
    """Multi-dimensional sinusoidal regression task.
    
    Learns functions of the form y = A * sin(x·ω + φ) where:
    - x is multi-dimensional input
    - ω is a frequency vector (one per dimension)
    - · denotes dot product
    - A is amplitude scalar
    - φ is phase scalar
    """
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1.0, freq_min=0.5, freq_max=2.0):
        super(SinusoidalRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.freq_min = freq_min
        self.freq_max = freq_max

        if pool_dict is None and seeds is None:
            # Sample parameters for each batch
            self.amplitude = torch.empty(self.b_size).uniform_(0.5, 1.5) * scale
            # Each dimension gets its own frequency component
            self.frequency = torch.empty(self.b_size, self.n_dims).uniform_(freq_min, freq_max)
            self.phase = torch.empty(self.b_size).uniform_(0, 2 * math.pi)
        elif seeds is not None:
            # Create parameters deterministically for each batch element
            self.amplitude = torch.zeros(self.b_size)
            self.frequency = torch.zeros(self.b_size, self.n_dims)
            self.phase = torch.zeros(self.b_size)
            
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.amplitude[i] = torch.empty(1, generator=generator).uniform_(0.5, 1.5) * scale
                self.frequency[i] = torch.empty(self.n_dims, generator=generator).uniform_(freq_min, freq_max)
                self.phase[i] = torch.empty(1, generator=generator).uniform_(0, 2 * math.pi)
        else:
            assert "amplitude" in pool_dict and "frequency" in pool_dict and "phase" in pool_dict
            indices = torch.randperm(len(pool_dict["amplitude"]))[:batch_size]
            self.amplitude = pool_dict["amplitude"][indices]
            self.frequency = pool_dict["frequency"][indices]
            self.phase = pool_dict["phase"][indices]

    def evaluate(self, xs_b):
        """
        Computes y = A * sin(x·ω + φ) where:
        - x·ω is the dot product between input and frequency vectors
        """
        # Move parameters to device
        amplitude = self.amplitude.to(xs_b.device).unsqueeze(1)  # Shape: [batch_size, 1]
        frequency = self.frequency.to(xs_b.device)  # Shape: [batch_size, n_dims]
        phase = self.phase.to(xs_b.device).unsqueeze(1)  # Shape: [batch_size, 1]
        
        # Compute dot product between input and frequency vectors
        # xs_b has shape [batch_size, n_points, n_dims]
        # frequency has shape [batch_size, n_dims]
        # We need to compute the dot product for each point
        
        # Reshape frequency for broadcasting: [batch_size, 1, n_dims]
        frequency = frequency.unsqueeze(1)
        
        # Compute dot product: [batch_size, n_points]
        dot_product = torch.sum(xs_b * frequency, dim=2)
        
        # Calculate sin(x·ω + φ)
        return amplitude * torch.sin(dot_product + phase)

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, scale=1.0, freq_min=0.5, freq_max=2.0, **kwargs):
        return {
            "amplitude": torch.empty(num_tasks).uniform_(0.5, 1.5) * scale,
            "frequency": torch.empty(num_tasks, n_dims).uniform_(freq_min, freq_max),
            "phase": torch.empty(num_tasks).uniform_(0, 2 * math.pi)
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class FourierFitBaseline:
    """
    Baseline: fits y = c0 + sum_{j=1}^n_dims sum_{k=1}^n_harmonics [a_{j,k} cos(k x_j) + b_{j,k} sin(k x_j)]
    to the data using least squares.
    """
    def __init__(self, n_dims, n_harmonics=3):
        self.n_dims = n_dims
        self.n_harmonics = n_harmonics
        self.coefs = None
        self.name = f"fourier_fit_n_dims={n_dims}_n_harmonics={n_harmonics}"

    def _fourier_features(self, xs):
        # xs: [N, n_dims] or [n_dims]
        # Ensure xs has the right shape for processing
        if xs.dim() == 1:
            # Single example: [n_dims] -> [1, n_dims]
            xs = xs.unsqueeze(0)
        
        feats = [torch.ones(xs.shape[0], 1, device=xs.device)]
        for j in range(self.n_dims):
            for k in range(1, self.n_harmonics + 1):
                feats.append(torch.sin(k * xs[:, j:j+1]))
                feats.append(torch.cos(k * xs[:, j:j+1]))
        return torch.cat(feats, dim=1)  # [N, 1 + 2*n_harmonics*n_dims]

    def fit(self, xs, ys):
        # xs: [N, n_dims], ys: [N]
        X = self._fourier_features(xs)
        # Use torch.linalg.lstsq for modern PyTorch
        self.coefs = torch.linalg.lstsq(X, ys.unsqueeze(1)).solution.squeeze(1)

    def predict(self, xs):
        X = self._fourier_features(xs)
        return X @ self.coefs
        
    def __call__(self, xs, ys, inds=None):
        """
        In-context learning style prediction. For each index i in inds:
        1. Fit the Fourier model on the first i points
        2. Predict for the i-th point
        """
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        
        preds = []
        
        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0])) # predict zero for first point
                continue
                
            batch_preds = []
            # Process each batch element individually
            for b in range(xs.shape[0]):
                # Fit model on points seen so far
                train_xs, train_ys = xs[b, :i], ys[b, :i]
                self.fit(train_xs, train_ys)
                
                # Predict for the current point
                test_x = xs[b, i]  # Get the actual point, not a slice
                pred = self.predict(test_x)
                batch_preds.append(pred.item() if isinstance(pred, torch.Tensor) else pred)
                
            preds.append(torch.tensor(batch_preds, device=xs.device))
            
        return torch.stack(preds, dim=1)


class KernelRidgeFixedBaseline:
    """
    Baseline: Kernel Ridge Regression using a fixed RFF kernel (w_rff, b_rff).
    Fits the optimal linear weights for the given kernel.
    """
    def __init__(self, w_rff, b_rff, alpha=1e-6):
        self.w_rff = w_rff  # [rff_dim, n_dims]
        self.b_rff = b_rff  # [rff_dim]
        self.alpha = alpha
        self.coefs = None

    def _rff_features(self, xs):
        # xs: [N, n_dims]
        # w_rff: [rff_dim, n_dims], b_rff: [rff_dim]
        # Output: [N, rff_dim]
        device = xs.device
        w_rff = self.w_rff.to(device)
        b_rff = self.b_rff.to(device)
        # [N, rff_dim]
        phi_x = torch.cos(xs @ w_rff.t() + b_rff)
        phi_x = math.sqrt(2.0 / w_rff.shape[0]) * phi_x
        return phi_x

    def fit(self, xs, ys):
        # xs: [N, n_dims], ys: [N]
        X = self._rff_features(xs)
        # Kernel ridge regression: (X^T X + alpha I)^{-1} X^T y
        n_feat = X.shape[1]
        reg = self.alpha * torch.eye(n_feat, device=X.device)
        XtX = X.t() @ X
        XtY = X.t() @ ys.unsqueeze(1)
        self.coefs = torch.linalg.solve(XtX + reg, XtY).squeeze(1)

    def predict(self, xs):
        X = self._rff_features(xs)
        return X @ self.coefs
