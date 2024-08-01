import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
from itertools import product
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT


def rgb2gray(rgb):
    """ Convert an RGB digital image to greyscale. """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def vectorization(img, Cr, Cc, renorm = False):
    """ Vectorize the image as follows. 
        1. Split the original (Mr, Mc) image into S equal-sized patches of 
            shape (Cr, Cc). Then, vectorize each patch and collect all 
            in a (S, Cr*Cc) array, called "vect_patches".
        2. Normalize each (Cr*Cc) vector to the intensity of the corresponding 
            (Cr, Cc) patch. When "renorm" is set to True, save the 
            normalization constants in the array "norm" for final decoding. 
            Otherwise, erase this information by setting "norm" equal to an 
            array of 1s.
        3. Define an array called "states" with shape (S, Cr*Cc), obtained as 
            the elementwise square root of "vect_patches".
    Return the couple (states, norm). """
    
    Mr, Mc = img.shape # Shape of the original image (#rows, #columns)
    
    # The image is split into N patches, each of shape (Cr,Cc)
    patches =  (img.reshape(Mc//Cr, Cr, -1, Cc).swapaxes(1, 2)\
                .reshape(-1, Cr, Cc)) # Shape (S, Cr, Cc)
    
    # Vectorization
    vect_patches = np.reshape(patches,\
                              (patches.shape[0],Cr*Cc)) # Shape (S, Cr*Cc)
    
    # Normalization
    states = np.zeros((patches.shape[0],Cr*Cc)) # Shape (S, Cr*Cc)
    
    norm = np.zeros(patches.shape[0]) # Shape (S, 1)
    for idx in range(patches.shape[0]):
        norm[idx] = vect_patches[idx].sum()
        if norm[idx] == 0:
            raise ValueError('Pixel value is 0') 
        tmp = vect_patches[idx]/norm[idx]
        states[idx] = np.sqrt(tmp)
    if renorm == False:
        norm = np.ones(patches.shape[0])
        
    return (states, norm)

# See https://qiskit.org/textbook/ch-algorithms/quantum-fourier-transform.html for further details about the
# implementations of these functions

def qft_rotations(circuit, n):
    """ QFT rotations on the first n qubits in circuit (without swaps). """
    if n == 0:
        return circuit
    n -= 1
    circuit.h(n)
    
    for qubit in range(n):
        circuit.cp(np.pi/2**(n-qubit), qubit, n)
    # At the end of the function, call it again on
    # the next qubits (reduced n by one earlier in the function)

def swap_registers(circuit, n):
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit

def qft(circuit, n):
    """ QFT on the first n qubits in circuit. """
    qft_rotations(circuit, n)
    swap_registers(circuit, n)
    return circuit

def iqft(circuit, n):
    """ IQFT on the first n qubits in circuit. """
    # First create a QFT circuit of the correct size
    
    qft_circ = qft(QuantumCircuit(n), n)
    
    # Take the inverse of this circuit
    invqft_circ = qft_circ.inverse()
    
    # Add it to the first n qubits in circuit
    circuit.append(invqft_circ, circuit.qubits[:n])
    
    return circuit

def circuit_builder(states, n0, n2):
    """ Create a list of n-qubit quantum circuits "qcs", each initialized 
    using the array "states". For each circuit
        1. Perform QFT
        2. Apply IQFT on the first (n0 - n2)//2 qubits
        3. Discard (n0 - n2)//2 qubits from the middle of the register, then 
            perform a measurement of the remaining qubits and store the 
            results in a classical register of n2 bits 
    Return "qcs". """
    
    ntilde = (n0 - n2)//2 # Total number of qubits removed aat each Step
    n1 =  n0 - ntilde # Number of qubits kept, before IQFT is applied
    
    qcs = []
    for idx in range(states.shape[0]):
        q = QuantumRegister(n0)
        c = ClassicalRegister(n2)
        qc = QuantumCircuit(q,c)
        
        qc.initialize(states[idx], q)    
        
        qc.h(q) # Optional, improves the reconstruction
        
        # Apply QFT on the full circuit
        qc.compose(QFT(num_qubits=n0, approximation_degree=0, do_swaps=True, \
                       inverse=False, insert_barriers=True, name='QFT'), \
                       inplace = True)
        
        qc.barrier()
        
        # Apply IQFT on the first n1 qubits (Rule 1)
        qc.compose(QFT(num_qubits=n1, approximation_degree=0, do_swaps=True, \
                       inverse=True, insert_barriers=True, name='IQFT'), \
                       qubits = q[0:n1], inplace = True)
        
        creg_idx = 0
        for idx in range(n1):  
            if n0//2 - ntilde <= idx <= n0//2 - 1: 
                continue # Qubit discarded from the measurement (Rule 2)
            qc.h(q[idx]) # Optional, improves the reconstruction
            qc.measure(q[idx],c[creg_idx])
            creg_idx += 1
        qcs.append(qc)
        
    return qcs

def reconstruction(qcs, n2, shots, norm):
    """ Simulate "qcs" with a given number of "shots". 
    Return an array "out_freq" with shape (S, 2**n2), describing the output 
    frequencies of each circuit and eventually rescaled using the corresponding
    component of "norm". """
    
    out_freq = np.zeros((len(qcs), 2**n2)) # Shape (S, 2**n2)
    
    for idx in range(len(qcs)):
        simulator = AerSimulator()
        qcs[idx] = transpile(qcs[idx], simulator)
        result = simulator.run(qcs[idx], shots = shots).result()
        
        counts = result.get_counts(qcs[idx]) # Counts at the output of qcs[idx]
        tot = sum(counts.values(), 0.0)
        prob = {key: val/tot for key, val in counts.items()} # Frequencies
        
        out = np.zeros(2**n2) # Shape (1, 2**n2)
        
        # Generate all the possible n2 qubits configurations
        cfgs = list(product(('0','1'), repeat = n2))
        cfgs = [''.join(cfg) for cfg in cfgs] 
        for i in range(2**n2):
            out[i] = prob.get(cfgs[i], 0)
            
        out_freq[idx,:] = out[:]*norm[idx]
    
    return out_freq

def devectorization(out_freq):
    """ Reconstruct an image using the simulation output. The function 
    operates as follows:
        1. Devectorize each of the S arrays in "out_freq" to 
            a (2**(n2/2), 2**(n2/2)) patch.
        2. Recombine the patches in a single object called "decoded_img" with 
        shape (Mr, Mc). 
    Return an image with number of pixels equal to the length of "out_freq" 
    times the number of patches. """
    
    S = out_freq.shape[0] # Number of patches
    nrow = int(np.sqrt(out_freq.shape[1])) # Number of rows of each patch
    ncol = nrow # Number of columns of each compressed patch
    
    decoded_patches = np.reshape(out_freq,\
                      (out_freq.shape[0], nrow, ncol)) # Shape (S, nrow, ncol)
    
    im_h, im_w = nrow*int(np.sqrt(S)), ncol*int(np.sqrt(S)) # Final shape
    
    decoded_img = np.zeros((im_w, im_h)) # Initialization
    
    idx = 0
    for row in np.arange(im_h - nrow + 1, step=nrow):
        for col in np.arange(im_w - ncol + 1, step=ncol):
            decoded_img[row:row+nrow, col:col+ncol] = decoded_patches[idx]
            idx += 1
            
    return decoded_img