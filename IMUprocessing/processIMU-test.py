## processIMU.py Test Script 
# Written By: EJ Rainville, Summer 2021

# Import Packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from termcolor import colored
print('Packages Loaded')

# Import Test Data and Test Output
test_data = sio.loadmat('IMUtoXYZ-testdata.mat')

# Organize data into each component from .mat file
# Accelarations
ax = test_data['ax']
ay = test_data['ay']
az = test_data['az']
# Gyroscope rotations
gx = test_data['gx']
gy = test_data['gy']
gz = test_data['gz']
# Magnetometer rotations
mx = test_data['mx']
my = test_data['my']
mz = test_data['mz']
# Magnetometer rotations offsets
mxo = test_data['mxo']
myo = test_data['myo']
mzo = test_data['mzo']
# Weight Factor and sampling frequency
Wd = test_data['Wd']
fs = test_data['fs']

# Define output from MATLAB function to compare to 
# all are suffized with _t for "true" value
x_t = test_data['x']
y_t = test_data['y']
z_t = test_data['z']
roll_t = test_data['roll']
pitch_t = test_data['pitch']
yaw_t = test_data['yaw']
heading_t = test_data['heading']

# Run IMUtoXYZ.py function
from IMUtoXYZ import IMUtoXYZ
x, y, z, roll, pitch, yaw, heading = IMUtoXYZ(ax, ay, az, gx, gy, gz, mx, my, mz, mxo, myo, mzo, Wd, fs)

# Test Output for the IMUtoXYZ function
precision = 1
# Test x
if(np.abs(np.linalg.norm(x)-np.linalg.norm(x_t)) <= precision ):
    print(colored('======== Test x Passed ========', 'green'))
else:
    print(colored('x Test Failed, norm(x) = ' + str(np.linalg.norm(x)) + ', norm(x_t) = ' + str(np.linalg.norm(x_t)), 'red'))

# Test y
if(np.abs(np.linalg.norm(y)-np.linalg.norm(y_t)) <= precision ):
    print(colored('======== Test y Passed ========', 'green'))
else:
    print(colored('y Test Failed, norm(x) = ' + str(np.linalg.norm(y)) + ', norm(t_t) = ' + str(np.linalg.norm(y_t)), 'red'))

# Test z
if(np.abs(np.linalg.norm(z)-np.linalg.norm(z_t)) <= precision ):
    print(colored('======== Test z Passed ========', 'green'))
else:
    print(colored('z Test Failed, norm(z) = ' + str(np.linalg.norm(z)) + ', norm(z_t) = ' + str(np.linalg.norm(z_t)), 'red'))

# Test roll
if(np.abs(np.linalg.norm(roll)-np.linalg.norm(roll_t)) <= precision ):
    print(colored('======== Test Roll Passed ========', 'green'))
else:
    print(colored('roll Test Failed, norm(roll) = ' + str(np.linalg.norm(roll)) + ', norm(roll_t) = ' + str(np.linalg.norm(roll_t)), 'red'))
    # Plot the data
    fig_roll, ax = plt.subplots()
    ax.plot(roll, label='roll - python')
    ax.plot(roll_t, label='roll - matlab')
    ax.legend()
    plt.show()

# Test pitch
if(np.abs(np.linalg.norm(pitch)-np.linalg.norm(pitch_t)) <= precision ):
    print(colored('======== Test pitch Passed ========', 'green'))
else:
    print(colored('pitch Test Failed, norm(pitch) = ' + str(np.linalg.norm(pitch)) + ', norm(pitch_t) = ' + str(np.linalg.norm(pitch_t)), 'red'))
    # Plot the data
    fig_pitch, ax = plt.subplots()
    ax.plot(pitch, label='pitch - python')
    ax.plot(pitch_t, label='pitch - matlab')
    ax.legend()
    plt.show()

# Test yaw
if(np.abs(np.linalg.norm(yaw)-np.linalg.norm(yaw_t)) <= precision ):
    print(colored('======== Test yaw Passed ========', 'green'))
else:
    print(colored('yaw Test Failed, norm(yaw) = ' + str(np.linalg.norm(yaw)) + ', norm(yaw_t) = ' + str(np.linalg.norm(yaw_t)), 'red'))

# Test heading
if(np.abs(np.linalg.norm(heading)-np.linalg.norm(heading_t)) <= precision ):
    print(colored('======== Test heading Passed ========', 'green'))
else:
    print(colored('heading Test Failed, norm(heading) = ' + str(np.linalg.norm(heading)) + ', norm(heading_t) = ' + str(np.linalg.norm(heading_t)), 'red'))
    # Plot the data
    fig_heading, ax = plt.subplots()
    ax.plot(heading, label='heading - python')
    ax.plot(heading_t, label='heading - matlab')
    ax.legend()
    plt.show()
    