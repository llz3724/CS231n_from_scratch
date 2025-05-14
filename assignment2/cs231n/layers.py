from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N=x.shape[0]
    M=w.shape[1]
    new_x=x.reshape((N,-1))
    new_b=b.reshape((1,M))
    out=new_x@w+new_b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    new_x=x.reshape((x.shape[0],-1))
    dw=new_x.T@dout
    dx=(dout@w.T).reshape(x.shape)
    db=dout.sum(axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out=np.maximum(0,x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    coeff=(x>0)
    dx=dout*coeff

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N=x.shape[0]
    C=x.shape[1]
    exp_x=np.exp(x)
    loss=(-x[np.arange(N),y].sum()+np.log(exp_x.sum(axis=1)).sum())*1/N

    dx=np.zeros((N,C))
    dx[np.arange(N),y]=-1
    sum_exp=exp_x.sum(axis=1)
    sum_exp=sum_exp.reshape((N,1))
    dx+=exp_x/sum_exp
    dx*=1/N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mu=x.mean(axis=0)
        variance=x.var(axis=0)
        std_x=(x-mu)/np.sqrt(variance+eps)
        out=gamma*std_x+beta

        running_mean=momentum*running_mean+(1-momentum)*mu
        running_var=momentum*running_var+(1-momentum)*variance

        cache=[mu,variance,std_x,x-mu,eps,gamma]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out=(x-running_mean)/np.sqrt(running_var+eps)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    mu,variance,std_x,x1,eps,gamma=cache
    N,D=std_x.shape
    dbeta=dout.sum(axis=0)
    dgamma=(dout*std_x).sum(axis=0)
    dstd_x=gamma*dout
    dvariance=(-0.5*dstd_x*std_x/(variance+eps)).sum(axis=0)
    dmu=-(dstd_x/np.sqrt(variance+eps)).sum(axis=0)
    dmu=dmu.reshape((1,-1))
    dvariance=dvariance.reshape((1,-1))
    dx=1.0/N*dmu+2.0/N*x1*dvariance+1.0/np.sqrt(variance+eps)*dstd_x


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    mu,variance,std_x,x1,eps,gamma=cache
    
    N,D=std_x.shape
    dbeta=dout.sum(axis=0)
    dgamma=(dout*std_x).sum(axis=0)
    dstd_x=gamma*dout

    sq_fac=np.sqrt(variance+eps)
    # dx1=1/sq_fac*(dstd_x-std_x*np.mean(dstd_x*std_x,axis=0))
    # dx=dx1-dx1.mean(axis=0)

    # Calculate dx using the single simplified formula
    # dx = (1/stddev) * [dstd_x - mean(dstd_x) - std_x * mean(dstd_x * std_x)]
    dx=(dstd_x-np.mean(dstd_x,axis=0)-std_x*np.mean(dstd_x*std_x,axis=0))/sq_fac


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    mu=x.mean(axis=1)
    variance=x.var(axis=1)
    mu=mu.reshape((-1,1))
    variance=variance.reshape((-1,1))
    std_x=(x-mu)/np.sqrt(variance+eps)
    out=gamma*std_x+beta
    cache=[mu,variance,std_x,x-mu,eps,gamma]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    mu,variance,std_x,x1,eps,gamma=cache
    
    N,D=std_x.shape
    dbeta=dout.sum(axis=0)
    dgamma=(dout*std_x).sum(axis=0)
    dstd_x=gamma*dout

    sq_fac=np.sqrt(variance+eps)
    sq_fac=sq_fac.reshape((-1,1))
    # dx1=1/sq_fac*(dstd_x-std_x*np.mean(dstd_x*std_x,axis=0))
    # dx=dx1-dx1.mean(axis=0)

    # Calculate dx using the single simplified formula
    # dx = (1/stddev) * [dstd_x - mean(dstd_x) - std_x * mean(dstd_x * std_x)]
    dx=(dstd_x-(np.mean(dstd_x,axis=1)).reshape((-1,1))-std_x*np.mean(dstd_x*std_x,axis=1).reshape((-1,1)))/sq_fac

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask=(np.random.rand(*x.shape)<p)/p
        out=x*mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out=x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx=dout*mask


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pad=conv_param['pad']
    stride=conv_param['stride']
    x_=np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant')

    N,C,H,W=x.shape
    F,C,HH,WW=w.shape
    nH=1+(H+2*pad-HH)//stride
    nW=1+(W+2*pad-WW)//stride
    out=np.empty((N,F,nH,nW))
    for i in range(N):
        for j in range(F):
            for row in range(nH):
                for col in range(nW):
                    start_h=row*stride
                    end_h=start_h+HH
                    start_w=col*stride
                    end_w=start_w+WW

                    x_patch=x_[i,:,start_h:end_h,start_w:end_w]
                    conv_sum=np.sum(w[j]*x_patch)
                    out[i,j,row,col]=conv_sum+b[j]
                    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x,w,b,conv_param=cache
    pad=conv_param['pad']
    stride=conv_param['stride']
    N,C,H,W=x.shape
    F,C,HH,WW=w.shape
    nH=1+(H+2*pad-HH)//stride
    nW=1+(W+2*pad-WW)//stride

    x_=np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant')

    db=np.empty(b.shape)
    for i in range(F):
        db[i]=(dout[:,i,:,:]).sum()
    
    
    dw=np.zeros(w.shape)
    for j in range(F):
        # x_[i,:,start_h:end_h,start_w:end_w]
        for i in range(N):
            for row in range(nH):
                for col in range(nW):
                    start_h=row*stride
                    end_h=start_h+HH
                    start_w=col*stride
                    end_w=start_w+WW
                    dw[j]+=dout[i,j,row,col]*x_[i,:,start_h:end_h,start_w:end_w]

    dx_=np.zeros(x_.shape)
    for j in range(F):
        # x_[i,:,start_h:end_h,start_w:end_w]
        for i in range(N):
            for row in range(nH):
                for col in range(nW):
                    start_h=row*stride
                    end_h=start_h+HH
                    start_w=col*stride
                    end_w=start_w+WW
                    dx_[i,:,start_h:end_h,start_w:end_w]+=dout[i,j,row,col]*w[j]
    dx=dx_[:,:,pad:H+pad,pad:W+pad]


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pool_height=pool_param['pool_height']
    pool_width=pool_param['pool_width']
    stride=pool_param['stride']

    N,C,H,W=x.shape
    nH=1+(H-pool_height)//stride
    nW=1+(W-pool_width)//stride
    out=np.empty((N,C,nH,nW))
    for i in range(N):
        for j in range(C):
            for row in range(nH):
                for col in range(nW):
                    start_h=row*stride
                    end_h=start_h+pool_height
                    start_w=col*stride
                    end_w=start_w+pool_width

                    x_patch=x[i,j,start_h:end_h,start_w:end_w]
                    out[i,j,row,col]=np.max(x_patch)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x,pool_param=cache
    pool_height=pool_param['pool_height']
    pool_width=pool_param['pool_width']
    stride=pool_param['stride']

    N,C,H,W=x.shape
    nH=1+(H-pool_height)//stride
    nW=1+(W-pool_width)//stride

    dx=np.zeros(x.shape)
    
    for i in range(N):
        for j in range(C):
            for row in range(nH):
                for col in range(nW):
                    start_h=row*stride
                    end_h=start_h+pool_height
                    start_w=col*stride
                    end_w=start_w+pool_width

                    x_patch=x[i,j,start_h:end_h,start_w:end_w]
                    
                    flat_idx=np.argmax(x_patch)
                    h_idx,w_idx=np.unravel_index(flat_idx,(pool_height,pool_width))
                    h_idx+=start_h
                    w_idx+=start_w
                    dx[i,j,h_idx,w_idx]+=dout[i,j,row,col]


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N,C,H,W=x.shape
    x_=(x.transpose((0,2,3,1))).reshape((-1,C)) #N,H,W,C
    out,cache=batchnorm_forward(x_,gamma,beta,bn_param)
    out=(out.reshape((N,H,W,C))).transpose((0,3,1,2))


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N,C,H,W=dout.shape
    dout_=(dout.transpose((0,2,3,1))).reshape((-1,C)) #N,H,W,C
    dx,dgamma,dbeta=batchnorm_backward_alt(dout_,cache)
    dx=(dx.reshape((N,H,W,C))).transpose((0,3,1,2))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N,C,H,W=x.shape
    x_=x.reshape(N,G,C//G,H,W)
    mu=x_.mean(axis=(2,3,4),keepdims=True)
    variance=x_.var(axis=(2,3,4),keepdims=True)
    std_x=((x_-mu)/np.sqrt(variance+eps)).reshape((N,C,H,W))
    out=gamma*std_x+beta
    # cache=[x,mu,variance,eps,std_x,gamma,G]
    ivar = 1./np.sqrt(variance + eps)
    x_grouped=x_
    x_hat=std_x
    cache = [x, x_hat, mu, variance, gamma, beta, G, eps, ivar, x_grouped]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # x,mu,variance,eps,std_x,gamma,G=cache
    # N,C,H,W=dout.shape

    # dbeta=dout.sum(axis=(0,2,3),keepdims=True)
    # dgamma=dout.sum(axis=(0,2,3),keepdims=True)
    # dout_reshaped=(dout.transpose((0,2,3,1))).reshape((-1,C))
    # dx_reshaped,dgamma_reshaped,dbeta_reshaped=

    x, x_hat, mu, variance, gamma, beta, G, eps, ivar, x_grouped = cache
    N, C, H, W = dout.shape
    C_per_group = C // G
    M = C_per_group * H * W # Number of elements per group = D' for LayerNorm view

    # 2. Calculate dbeta and dgamma (must be done based on GroupNorm structure)
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)    # Shape (1, C, 1, 1)
    dgamma = np.sum(dout * x_hat, axis=(0, 2, 3), keepdims=True) # Shape (1, C, 1, 1)

    # 3. Prepare inputs for layernorm_backward

    # Reshape dout: (N, C, H, W) -> (N, G, Cg, H, W) -> (N*G, Cg*H*W)
    # Pre-scale dout by gamma before passing to layernorm_backward
    dout_scaled = dout * gamma # Shape (N, C, H, W) using broadcasting
    dout_ln = dout_scaled.reshape(N, G, C_per_group, H, W).transpose(0, 1, 2, 3, 4).reshape(N * G, M)

    # Reshape intermediates for the layernorm cache
    # x_hat (std_x for layernorm) : (N, C, H, W) -> (N*G, Cg*H*W)
    x_hat_ln = x_hat.reshape(N, G, C_per_group, H, W).transpose(0, 1, 2, 3, 4).reshape(N * G, M)
    # mu : (N, G, 1, 1, 1) -> (N*G, 1)
    mu_ln = mu.reshape(N * G, 1)
    # variance : (N, G, 1, 1, 1) -> (N*G, 1)
    var_ln = variance.reshape(N * G, 1)
    # eps: scalar
    # gamma for layernorm: Needs shape (D',). Since we pre-scaled dout, use ones.
    gamma_ln = np.ones(M)
    # x1: Not directly available or used in simplified formula, pass None
    x1_ln = None # Or potentially a reshaped x if layernorm needed it

    # Create the cache tuple expected by layernorm_backward
    # Assuming format: (mu, variance, std_x, x1, eps, gamma)
    cache_ln = (mu_ln, var_ln, x_hat_ln, x1_ln, eps, gamma_ln)

    # 4. Call layernorm_backward
    # Pass the pre-scaled dout (dout_ln) and the constructed cache
    # We only need dx_ln from the result
    dx_ln, _, _ = layernorm_backward(dout_ln, cache_ln) # Ignore dgamma_ln, dbeta_ln

    # 5. Reshape dx_ln back to the original input shape
    # (N*G, Cg*H*W) -> (N, G, Cg, H, W) -> (N, C, H, W)
    dx = dx_ln.reshape(N, G, C_per_group, H, W).transpose(0, 1, 2, 3, 4).reshape(N, C, H, W)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
