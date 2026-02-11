
import numpy as np

def random_polarity(nsrc):
    """
    Generate a polarity vector for each shot.
    
    Returns
    -------
    np.ndarray
        Array of shape (nsrc,) containing -1 or +1 for each shot.
    """
    return np.random.choice([-1, 1], size=nsrc)

def random_time_delay(nsrc, nt, delay):
    """
    Generate a shift matrix for each shot, for random time delay encoding.
    
    Parameters
    ----------
    nsrc : int
        Number of shots.
    nt : int
        Number of time samples.
    delay : float
        Maximum delay as a fraction of nt (e.g., 0.2 means up to 20% of nt).
    
    Returns
    -------
    np.ndarray
        Array of shape (nsrc, nt, nt) where each [i,:,:] is a shift matrix.
    """
    max_delay = int(delay * nt)
    delays = np.random.randint(0, max_delay + 1, size=nsrc)
    # nt = int(nt* (1 + delay))
    shift_matrices = np.zeros((nsrc, nt, nt))
    for i in range(nsrc):
        d = delays[i]
        I = np.eye(nt)
        if d > 0:
            shift_matrices[i, d:, :] = I[:-d, :]
        else:
            shift_matrices[i] = I
    return shift_matrices

class Encoding:
    """
    Class for applying source encoding.

    Parameters
    ----------
    encoding : str
        Encoding type. Supported values are 'random_polarity' and 'random_time_delay'.
    nsrc : int
        Number of shots.
    nt : int, optional
        Number of time samples. Required if encoding is 'random_time_delay'.
    delay : float, optional
        Maximum delay as a fraction of nt (default: 0.2), used for 'random_time_delay'.
    """
    def __init__(self, encoding, nsrc, encoding_params):
        self.encoding = encoding
        self.nsrc = nsrc
        self.encoding_params = encoding_params
        self.enc = self._generate_encoding()

    def _generate_encoding(self):
        if self.encoding == 'random_polarity':
            return random_polarity(self.nsrc)
        elif self.encoding == 'random_time_delay':
            nt = self.encoding_params.get('nt')
            delay = self.encoding_params.get('delay', 0.2)
            return random_time_delay(self.nsrc, nt, delay)
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}. Supported encodings: 'random_polarity', 'random_time_delay'.")

    def apply(self, data, composite=True):
        """
        Apply the encoding to the input data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data of shape (nsrc, nt, ...). For 'random_polarity' encoding, each shot is multiplied
            by its corresponding scalar. For 'random_time_delay' encoding, each shot is transformed by
            matrix-multiplying along the time dimension.
        composite : bool, optional
            If True and data.ndim == 3, return the sum over the first dimension (i.e. composite response).
        
        Returns
        -------
        np.ndarray
            The encoded data with the same shape as the input, or composite if composite is True.
        """
        # Ensure data has shape (nsrc, nt, ...)
        # if data.shape[0] != self.nsrc and data.shape[1] == self.nsrc:
            # If the first dimension is not equal to nsrc, assume the axes are swapped.
            # data = data.T
        enc_data = np.empty_like(data)
        # print(enc_data.shape, self.enc.shape, data.shape)
        if self.encoding == 'random_polarity':
            for i in range(self.nsrc):
                enc_data[i] = self.enc[i] * data[i]
        elif self.encoding == 'random_time_delay':
            for i in range(self.nsrc):
                # print(i)
                # print(enc_data[i].shape, self.enc[i].shape, data[i].shape)
                enc_data[i] = self.enc[i] @ data[i]
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}.")
        
        if composite and data.ndim == 3:
            return enc_data.sum(axis=0)
        return enc_data

    def regenerate(self):
        """
        Regenerate the encoding.
        """
        self.enc = self._generate_encoding()
        
    def __call__(self, data, composite=False):
        return self.apply(data, composite=composite)

# import numpy as np

# def random_polarity(nsrc):
#     """
#     Generate a polarity vector for each shot.
    
#     Returns
#     -------
#     np.ndarray
#         Array of shape (nsrc,) containing -1 or +1 for each shot.
#     """
#     return np.random.choice([-1, 1], size=nsrc)

# def random_time_delay(nsrc, nt, delay):
#     """
#     Generate a shift matrix for each shot, for random time delay encoding.
    
#     Parameters
#     ----------
#     nsrc : int
#         Number of shots.
#     nt : int
#         Number of time samples.
#     delay : float
#         Maximum delay as a fraction of nt (e.g., 0.2 means up to 20% of nt).
    
#     Returns
#     -------
#     np.ndarray
#         Array of shape (nsrc, nt, nt) where each [i,:,:] is a shift matrix.
#     """
#     max_delay = int(delay * nt)
#     # Uniform random delays between 0 and max_delay.
#     delays = np.random.randint(0, max_delay + 1, size=nsrc)
#     print("Delays:", delays, flush=True)
#     shift_matrices = []
#     for isrc in range(nsrc):
#         time_delay = delays[isrc]
#         shift_matrix = np.zeros((nt, nt))
#         for ii in range(nt - time_delay):
#             shift_matrix[ii + time_delay, ii] = 1
#         shift_matrices.append(shift_matrix)
#     return np.array(shift_matrices)

# class Encoding:
#     """
#     Class for applying source encoding.

#     Parameters
#     ----------
#     encoding : str
#         Encoding type. Supported values are 'random_polarity' and 'random_time_delay'.
#     nsrc : int
#         Number of shots.
#     nt : int, optional
#         Number of time samples. Required if encoding is 'random_time_delay'.
#     delay : float, optional
#         Maximum delay as a fraction of nt (default: 0.2), used for 'random_time_delay'.
#     """
#     def __init__(self, encoding, nsrc, nt=None, delay=0.2):
#         self.encoding = encoding
#         self.nsrc = nsrc
#         self.nt = nt
#         self.delay = delay
#         self.enc = self._generate_encoding()

#     def _generate_encoding(self):
#         if self.encoding == 'random_polarity':
#             return random_polarity(self.nsrc)
#         elif self.encoding == 'random_time_delay':
#             if self.nt is None:
#                 raise ValueError("nt must be provided for random_time_delay encoding.")
#             return random_time_delay(self.nsrc, self.nt, self.delay)
#         else:
#             raise ValueError(f"Unknown encoding: {self.encoding}. Supported encodings: 'random_polarity', 'random_time_delay'.")

#     def apply(self, data):
#         """
#         Apply the encoding to the input data.
        
#         Parameters
#         ----------
#         data : np.ndarray
#             Input data of shape (nsrc, nt, ...). For 'random_polarity' encoding, each shot is multiplied
#             by its corresponding scalar. For 'random_time_delay' encoding, each shot is transformed by
#             matrix-multiplying along the time dimension.
        
#         Returns
#         -------
#         np.ndarray
#             The encoded data with the same shape as the input.
#         """
#         enc_data = np.empty_like(data)
#         print(self.enc)
#         if self.encoding == 'random_polarity':
#             # self.enc is shape (nsrc,)
#             for i in range(self.nsrc):
#                 enc_data[i] = self.enc[i] * data[i]
#         elif self.encoding == 'random_time_delay':
#             # self.enc is shape (nsrc, nt, nt); data[i] should have shape (nt, ...)
#             for i in range(self.nsrc):
#                 enc_data[i] = self.enc[i] @ data[i]
#         else:
#             # Should not reach here because _generate_encoding() would have raised an error.
#             raise ValueError(f"Unknown encoding: {self.encoding}.")
        
#         if data.ndim == 3:
#             return enc_data.sum(0)
#         return enc_data

#     def __call__(self, data):
#         return self.apply(data)
