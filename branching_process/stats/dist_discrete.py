
try:
    import autograd
    import autograd.numpy as np
    import autograd.scipy as sp
    from autograd.scipy.special import gammaln
    have_autograd = True
except ImportError as e:
    import numpy as np
    import scipy as sp
    from scipy.special import gammaln
    have_autograd = False



def pmf_geom(k, mu):
    """Create exponential distribution likelihood subgraph
    :type mu: kernel param(scalar)
    :param mu: predicted points as column

    :type k: numpy.array
    :param k: eval points as column

    :return: graph of likelihood
    :rtype: numpy.array
    """
    return (k>0) * np.exp(
        loglik_geom(mu, eta, k))

def loglik_geom(k, mu, eta):
    """Create exponential distribution loglikelihood subgraph
    :type mu: kernel param(scalar)
    :param mu: predicted points as column

    :type k: numpy.array
    :param k: eval points as column

    :return: graph of loglihood
    :rtype: numpy.array
    """
    return k * np.log(mu) - (k + 1) * np.log(mu+1)


def pmf_gpd(mu, eta, k):
    """Create GPD distribution likelihood subgraph
    :type k: numpy.array
    :param k: eval points as column

    :type mu: float
    :param mu: distribution param: mean

    :type eta: float
    :param eta: distribution param: branch rate

    :return: graph of likelihood
    :rtype: numpy.array
    """
    return np.exp(loglik_gpd(mu, eta, k))


def loglik_gpd(mu, eta, k):
    """
    log-likelihood graph for GPD,
    $$P(X=k)=\frac{\mu(\mu+ \eta k)^{k-1}}{k!e^{\mu+k\eta}}$$

    :type mu: float
    :param mu: distribution param: mean

    :type eta: float
    :param eta: distribution param: branch rate

    :type k: numpy.array
    :param k: eval points as column

    :return: graph of loglihood
    :rtype: numpy.array
    """
    eta_k = eta * k
    mu_eta_k = mu + eta_k
    return np.log(mu) + (k-1)*np.log(mu_eta_k) - gammaln(k+1) - mu_eta_k


def pmf_poisson(mu, eta, k):
    """Create Poisson distribution likelihood subgraph
    :type mu: float
    :param mu: distribution param: mean

    :type k: numpy.array
    :param k: eval points as column

    :return: graph of likelihood
    :rtype: numpy.array
    """
    return np.exp(loglik_poisson(mu, eta, k))


def loglik_poisson(mu, k):
    """
    log-likelihood graph for Poisson counts. ``eta`` is ignored.
    $$P(X=k)=\frac{\mu^k}{k!e^\mu}$$

    :type mu: float
    :param mu: distribution param: mean
    :type k: numpy.array
    :param k: eval points as column

    :return: graph of loglihood
    :rtype: numpy.array
    """
    return k*np.log(mu) - gammaln(k+1) - mu

def pmf_polya(mu, alpha, k):
    """Create Polya distribution likelihood subgraph
    :type mu: float
    :param mu: distribution param: mean

    :type alpha: float
    :param alpha: distribution param: dispersion

    :type k: numpy.array
    :param k: eval points as column

    :return: graph of likelihood
    :rtype: numpy.array
    """
    return np.exp(loglik_polya(mu, eta, k))


def loglik_polya(mu, alpha, k):
    """
    log-likelihood graph for Polya counts.
    $$P(X=k)=\frac{\mu^k}{k!e^\mu}$$

    :type mu: float
    :param mu: distribution param: mean

    :type alpha: float
    :param alpha: distribution param: dispersion

    :type k: numpy.array
    :param k: eval points as column

    :return: graph of loglihood
    :rtype: numpy.array
    """
    return k*np.log(mu) - gammaln(k+1) - mu


def conv_conv2d(counts, rev_phi):
    """
    1d convolution in Tensorflow.
    This could be bumped up to 2d and used, but I won't because the padding is
    too complicated and weird.
    data shape is "[batch, in_height, in_width, in_channels]",
    and must be a float to be convolved with floats
    It must be float32 to be convolved at all.
    filter shape is "[filter_height, filter_width, in_channels, out_channels]"

    https://www.tensorflow.org/versions/r0.9/api_docs/python/nn.html#convolution

    #This doesn't quite have the right alignment.
    # Can I fix this by padding manually then requiring mode="VALID"?

    >>> from src import tf_graph_discrete
    >>> tf_graph_discrete = reload(tf_graph_discrete)
    >>> g = np.Graph()
    >>> counts_f = counts.astype("float32")
    >>> rev_kernel = geom_phi[::-1].astype("float32")
    >>> with g.as_default():
    >>>     counts_t = np.Variable(counts_f, name = "counts")
    >>>     rev_phi = np.Variable(rev_kernel, name="phi")
    >>>     conv_t = tf_graph_discrete.conv_conv2d(counts_t, rev_phi)
    >>> with np.Session(graph=g) as session:
    >>>     init_op =  np.initialize_all_variables()
    >>>     session.run([init_op])
    >>>     convd = conv_t.eval(session=session)
    >>> plt.plot(counts);
    >>> plt.plot(convd.ravel());
    """
    counts_t = np.reshape(counts, (1, 1, -1, 1))
    rev_phi_t = np.reshape(rev_phi, (1, -1, 1, 1))
    return np.nn.conv2d(
        counts_t,
        rev_phi_t,
        strides=[1, 1, 1, 1],
        padding="SAME",
        name="conv")

def conv_manual(
        counts_t,
        rev_phi_basis_t,
        ts_len,
        kernel_len,
        name='manual_conv'):
    with np.name_scope(name):
        partials = []
        for timestep in range(1, ts_len):
            kernel_start = max(0, kernel_len-timestep)
            ts_start = max(0, timestep-kernel_len)
            partials.append(np.reduce_sum(
                # np.transpose(
                    np.dot(
                        rev_phi_basis_t[:,kernel_start:kernel_len],
                        counts_t[:,ts_start:timestep],
                        transpose_b=True
                    )
                # )
                ,
                1
            ))
        conv_t = np.pack(partials)
    return conv_t
