# Signal Filter Tools from Scipy 
import numpy as np

# Detrend function
def detrend(data, axis=-1, type='linear', bp=0, overwrite_data=False):
    """
    Remove linear trend along axis from data.
    Parameters
    ----------
    data : array_like
        The input data.
    axis : int, optional
        The axis along which to detrend the data. By default this is the
        last axis (-1).
    type : {'linear', 'constant'}, optional
        The type of detrending. If ``type == 'linear'`` (default),
        the result of a linear least-squares fit to `data` is subtracted
        from `data`.
        If ``type == 'constant'``, only the mean of `data` is subtracted.
    bp : array_like of ints, optional
        A sequence of break points. If given, an individual linear fit is
        performed for each part of `data` between two break points.
        Break points are specified as indices into `data`. This parameter
        only has an effect when ``type == 'linear'``.
    overwrite_data : bool, optional
        If True, perform in place detrending and avoid a copy. Default is False
    Returns
    -------
    ret : ndarray
        The detrended input data.
    Examples
    --------
    >>> from scipy import signal
    >>> from numpy.random import default_rng
    >>> rng = default_rng()
    >>> npoints = 1000
    >>> noise = rng.standard_normal(npoints)
    >>> x = 3 + 2*np.linspace(0, 1, npoints) + noise
    >>> (signal.detrend(x) - noise).max()
    0.06  # random
    """
    if type not in ['linear', 'l', 'constant', 'c']:
        raise ValueError("Trend type must be 'linear' or 'constant'.")
    data = np.asarray(data)
    dtype = data.dtype.char
    if dtype not in 'dfDF':
        dtype = 'd'
    if type in ['constant', 'c']:
        ret = data - np.mean(data, axis, keepdims=True)
        return ret
    else:
        dshape = data.shape
        N = dshape[axis]
        bp = np.sort(np.unique(np.r_[0, bp, N]))
        if np.any(bp > N):
            raise ValueError("Breakpoints must be less than length "
                             "of data along given axis.")
        Nreg = len(bp) - 1
        # Restructure data so that axis is along first dimension and
        #  all other dimensions are collapsed into second dimension
        rnk = len(dshape)
        if axis < 0:
            axis = axis + rnk
        newdims = np.r_[axis, 0:axis, axis + 1:rnk]
        newdata = np.reshape(np.transpose(data, tuple(newdims)),
                             (N, _prod(dshape) // N))
        if not overwrite_data:
            newdata = newdata.copy()  # make sure we have a copy
        if newdata.dtype.char not in 'dfDF':
            newdata = newdata.astype(dtype)
        # Find leastsq fit and remove it for each piece
        for m in range(Nreg):
            Npts = bp[m + 1] - bp[m]
            A = np.ones((Npts, 2), dtype)
            A[:, 0] = np.cast[dtype](np.arange(1, Npts + 1) * 1.0 / Npts)
            sl = slice(bp[m], bp[m + 1])
            coef, resids, rank, s = linalg.lstsq(A, newdata[sl])
            newdata[sl] = newdata[sl] - np.dot(A, coef)
        # Put data back in original shape.
        tdshape = np.take(dshape, newdims, 0)
        ret = np.reshape(newdata, tuple(tdshape))
        vals = list(range(1, rnk))
        olddims = vals[:axis] + [0] + vals[axis:]
        ret = np.transpose(ret, tuple(olddims))
        return ret

# Second order filter
def sosfilt(sos, x, axis=-1, zi=None):
    """
    Filter data along one dimension using cascaded second-order sections.

    Filter a data sequence, `x`, using a digital IIR filter defined by
    `sos`.

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. Each row corresponds to a second-order
        section, with the first three columns providing the numerator
        coefficients and the last three providing the denominator
        coefficients.
    x : array_like
        An N-dimensional input array.
    axis : int, optional
        The axis of the input data array along which to apply the
        linear filter. The filter is applied to each subarray along
        this axis.  Default is -1.
    zi : array_like, optional
        Initial conditions for the cascaded filter delays.  It is a (at
        least 2D) vector of shape ``(n_sections, ..., 2, ...)``, where
        ``..., 2, ...`` denotes the shape of `x`, but with ``x.shape[axis]``
        replaced by 2.  If `zi` is None or is not given then initial rest
        (i.e. all zeros) is assumed.
        Note that these initial conditions are *not* the same as the initial
        conditions given by `lfiltic` or `lfilter_zi`.

    Returns
    -------
    y : ndarray
        The output of the digital filter.
    zf : ndarray, optional
        If `zi` is None, this is not returned, otherwise, `zf` holds the
        final filter delay values.

    See Also
    --------
    zpk2sos, sos2zpk, sosfilt_zi, sosfiltfilt, sosfreqz

    Notes
    -----
    The filter function is implemented as a series of second-order filters
    with direct-form II transposed structure. It is designed to minimize
    numerical precision errors for high-order filters.

    .. versionadded:: 0.16.0

    Examples
    --------
    Plot a 13th-order filter's impulse response using both `lfilter` and
    `sosfilt`, showing the instability that results from trying to do a
    13th-order filter in a single stage (the numerical error pushes some poles
    outside of the unit circle):

    >>> import matplotlib.pyplot as plt
    >>> from scipy import signal
    >>> b, a = signal.ellip(13, 0.009, 80, 0.05, output='ba')
    >>> sos = signal.ellip(13, 0.009, 80, 0.05, output='sos')
    >>> x = signal.unit_impulse(700)
    >>> y_tf = signal.lfilter(b, a, x)
    >>> y_sos = signal.sosfilt(sos, x)
    >>> plt.plot(y_tf, 'r', label='TF')
    >>> plt.plot(y_sos, 'k', label='SOS')
    >>> plt.legend(loc='best')
    >>> plt.show()

    """
    x = _validate_x(x)
    sos, n_sections = _validate_sos(sos)
    x_zi_shape = list(x.shape)
    x_zi_shape[axis] = 2
    x_zi_shape = tuple([n_sections] + x_zi_shape)
    inputs = [sos, x]
    if zi is not None:
        inputs.append(np.asarray(zi))
    dtype = np.result_type(*inputs)
    if dtype.char not in 'fdgFDGO':
        raise NotImplementedError("input type '%s' not supported" % dtype)
    if zi is not None:
        zi = np.array(zi, dtype)  # make a copy so that we can operate in place
        if zi.shape != x_zi_shape:
            raise ValueError('Invalid zi shape. With axis=%r, an input with '
                             'shape %r, and an sos array with %d sections, zi '
                             'must have shape %r, got %r.' %
                             (axis, x.shape, n_sections, x_zi_shape, zi.shape))
        return_zi = True
    else:
        zi = np.zeros(x_zi_shape, dtype=dtype)
        return_zi = False
    axis = axis % x.ndim  # make positive
    x = np.moveaxis(x, axis, -1)
    zi = np.moveaxis(zi, [0, axis + 1], [-2, -1])
    x_shape, zi_shape = x.shape, zi.shape
    x = np.reshape(x, (-1, x.shape[-1]))
    x = np.array(x, dtype, order='C')  # make a copy, can modify in place
    zi = np.ascontiguousarray(np.reshape(zi, (-1, n_sections, 2)))
    sos = sos.astype(dtype, copy=False)
    _sosfilt(sos, x, zi)
    x.shape = x_shape
    x = np.moveaxis(x, -1, axis)
    if return_zi:
        zi.shape = zi_shape
        zi = np.moveaxis(zi, [-2, -1], [0, axis + 1])
        out = (x, zi)
    else:
        out = x
    return out

# Butterworth iirfilter filter function
def iirfilter(N, Wn, rp=None, rs=None, btype='band', analog=False,
              ftype='butter', output='ba', fs=None):
    """
    IIR digital and analog filter design given order and critical points.

    Design an Nth-order digital or analog filter and return the filter
    coefficients.

    Parameters
    ----------
    N : int
        The order of the filter.
    Wn : array_like
        A scalar or length-2 sequence giving the critical frequencies.

        For digital filters, `Wn` are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`Wn` is thus in
        half-cycles / sample.)

        For analog filters, `Wn` is an angular frequency (e.g., rad/s).
    rp : float, optional
        For Chebyshev and elliptic filters, provides the maximum ripple
        in the passband. (dB)
    rs : float, optional
        For Chebyshev and elliptic filters, provides the minimum attenuation
        in the stop band. (dB)
    btype : {'bandpass', 'lowpass', 'highpass', 'bandstop'}, optional
        The type of filter.  Default is 'bandpass'.
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    ftype : str, optional
        The type of IIR filter to design:

            - Butterworth   : 'butter'
            - Chebyshev I   : 'cheby1'
            - Chebyshev II  : 'cheby2'
            - Cauer/elliptic: 'ellip'
            - Bessel/Thomson: 'bessel'

    output : {'ba', 'zpk', 'sos'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        second-order sections ('sos'). Default is 'ba' for backwards
        compatibility, but 'sos' should be used for general-purpose filtering.
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output=='sos'``.

    See Also
    --------
    butter : Filter design using order and critical points
    cheby1, cheby2, ellip, bessel
    buttord : Find order and critical points from passband and stopband spec
    cheb1ord, cheb2ord, ellipord
    iirdesign : General filter design using passband and stopband spec

    Notes
    -----
    The ``'sos'`` output parameter was added in 0.16.0.

    Examples
    --------
    Generate a 17th-order Chebyshev II analog bandpass filter from 50 Hz to
    200 Hz and plot the frequency response:

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

>>> b, a = signal.iirfilter(17, [2*np.pi*50, 2*np.pi*200], rs=60,
    ...                         btype='band', analog=True, ftype='cheby2')
    >>> w, h = signal.freqs(b, a, 1000)
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(1, 1, 1)
    >>> ax.semilogx(w / (2*np.pi), 20 * np.log10(np.maximum(abs(h), 1e-5)))
    >>> ax.set_title('Chebyshev Type II bandpass frequency response')
    >>> ax.set_xlabel('Frequency [Hz]')
    >>> ax.set_ylabel('Amplitude [dB]')
    >>> ax.axis((10, 1000, -100, 10))
    >>> ax.grid(which='both', axis='both')
    >>> plt.show()

    Create a digital filter with the same properties, in a system with
    sampling rate of 2000 Hz, and plot the frequency response. (Second-order
    sections implementation is required to ensure stability of a filter of
    this order):

    >>> sos = signal.iirfilter(17, [50, 200], rs=60, btype='band',
    ...                        analog=False, ftype='cheby2', fs=2000,
    ...                        output='sos')
    >>> w, h = signal.sosfreqz(sos, 2000, fs=2000)
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(1, 1, 1)
    >>> ax.semilogx(w, 20 * np.log10(np.maximum(abs(h), 1e-5)))
    >>> ax.set_title('Chebyshev Type II bandpass frequency response')
    >>> ax.set_xlabel('Frequency [Hz]')
    >>> ax.set_ylabel('Amplitude [dB]')
    >>> ax.axis((10, 1000, -100, 10))
    >>> ax.grid(which='both', axis='both')
    >>> plt.show()

    """
    ftype, btype, output = [x.lower() for x in (ftype, btype, output)]
    Wn = asarray(Wn)
    if fs is not None:
        if analog:
            raise ValueError("fs cannot be specified for an analog filter")
        Wn = 2*Wn/fs

    try:
        btype = band_dict[btype]
    except KeyError:
        raise ValueError("'%s' is an invalid bandtype for filter." % btype)

    try:
        typefunc = filter_dict[ftype][0]
    except KeyError:
        raise ValueError("'%s' is not a valid basic IIR filter." % ftype)

    if output not in ['ba', 'zpk', 'sos']:
        raise ValueError("'%s' is not a valid output form." % output)

    if rp is not None and rp < 0:
        raise ValueError("passband ripple (rp) must be positive")

    if rs is not None and rs < 0:
        raise ValueError("stopband attenuation (rs) must be positive")

    # Get analog lowpass prototype
    if typefunc == buttap:
        z, p, k = typefunc(N)
    elif typefunc == besselap:
        z, p, k = typefunc(N, norm=bessel_norms[ftype])
    elif typefunc == cheb1ap:
        if rp is None:
            raise ValueError("passband ripple (rp) must be provided to "
                             "design a Chebyshev I filter.")
        z, p, k = typefunc(N, rp)
    elif typefunc == cheb2ap:
        if rs is None:
            raise ValueError("stopband attenuation (rs) must be provided to "
                             "design an Chebyshev II filter.")
        z, p, k = typefunc(N, rs)
    elif typefunc == ellipap:
        if rs is None or rp is None:
            raise ValueError("Both rp and rs must be provided to design an "
                             "elliptic filter.")
        z, p, k = typefunc(N, rp, rs)
    else:
        raise NotImplementedError("'%s' not implemented in iirfilter." % ftype)

    # Pre-warp frequencies for digital filter design
    if not analog:
        if numpy.any(Wn <= 0) or numpy.any(Wn >= 1):
            if fs is not None:
                raise ValueError("Digital filter critical frequencies "
                                 "must be 0 < Wn < fs/2 (fs={} -> fs/2={})".format(fs, fs/2))
            raise ValueError("Digital filter critical frequencies "
                             "must be 0 < Wn < 1")
        fs = 2.0
        warped = 2 * fs * tan(pi * Wn / fs)
    else:
        warped = Wn

    # transform to lowpass, bandpass, highpass, or bandstop
    if btype in ('lowpass', 'highpass'):
        if numpy.size(Wn) != 1:
            raise ValueError('Must specify a single critical frequency Wn for lowpass or highpass filter')

        if btype == 'lowpass':
            z, p, k = lp2lp_zpk(z, p, k, wo=warped)
        elif btype == 'highpass':
            z, p, k = lp2hp_zpk(z, p, k, wo=warped)
    elif btype in ('bandpass', 'bandstop'):
        try:
            bw = warped[1] - warped[0]
            wo = sqrt(warped[0] * warped[1])
        except IndexError:
            raise ValueError('Wn must specify start and stop frequencies for bandpass or bandstop filter')

        if btype == 'bandpass':
            z, p, k = lp2bp_zpk(z, p, k, wo=wo, bw=bw)
        elif btype == 'bandstop':
            z, p, k = lp2bs_zpk(z, p, k, wo=wo, bw=bw)
    else:
        raise NotImplementedError("'%s' not implemented in iirfilter." % btype)

    # Find discrete equivalent if necessary
    if not analog:
        z, p, k = bilinear_zpk(z, p, k, fs=fs)

    # Transform to proper out type (pole-zero, state-space, numer-denom)
    if output == 'zpk':
        return z, p, k
    elif output == 'ba':
            return zpk2tf(z, p, k)
    elif output == 'sos':
        return zpk2sos(z, p, k)