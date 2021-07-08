# IMUtoXYZ
from matplotlib.pyplot import thetagrids
from numpy import deg2rad


def IMUtoXYZ(ax, ay, az, gx, gy, gz, mx, my, mz, mxo, myo, mzo, Wd, fs):
    """
    Function to calculate wave displacements in earth reference frame 
    from microSWIFT IMU measurements in body reference frame. 

    Adapted to native python version by EJ Rainville, Summer 2021
    from J. Thomsons original MATLAB version

    Parameters
    ----------
    ax, ay, az - raw accelerometer accelerations
    gx, gy, gz - raw gyroscope rotations
    mx, my, mz - raw magnetometer rotations
    mxo, yo, mzo - magnetometer offset values
    Wd - Weight coefficient for filtering the gyro
    fs - sampling frequency

    Returns
    -------
    x, y, z - displacement values
    roll, pitch, yaw - angles 
    heading

    The body reference frame for the inputs is
    x: along bottle (towards cap), roll around this axis
    y: accross bottle (right hand sys), pitch around this axis
    z: up (skyward, same as GPS), yaw around this axis

    """
    # Import Statements
    import numpy as np
    from scipy import integrate

    # Weighting from (0 to 1) for static angles in complimentary filter
    Ws = 1 - Wd

    # Define Time Step 
    dt = 1/fs

    # Define high-pass RC filter constant, T > (2* pi * RC)
    RC = 4

    # ---- Estimate Euler Angles ------------
    # Static Angles
    staticroll =  np.mean( np.rad2deg( np.arctan2(ay, az) ) ) # Roll around x-axis [degrees] in absence of linear acceleration
    staticpitch =  np.mean( np.rad2deg( np.arctan2(-ax, np.sqrt(ay**2, az**2) ) ) ) # Pich around y-axis [deg] in absence of linear accelerations
    staticyaw = 0 # zero until corrected by magnetometer

    # Define RC filter function
    def RCfilter(b, RC, fs):
        alpha = RC / (RC + 1/fs)
        a = b.copy()
        for n in np.arange(1, b.shape[0]):
            a[n] = (alpha * a[n-1]) + (alpha * (b[n] - b[n-1] ))
        return a

    # Dynamic Angles
    if Wd != 0:
        dynamicroll_unfilt = integrate.cumtrapz(gx, dx=dt, axis=0) # time integrate x rotations to get dynamic roll
        dynamicroll = RCfilter(dynamicroll_unfilt, RC, fs)
        dynamicpitch_unfilt = integrate.cumtrapz(gy, dx=dt, axis=0) # time integrate y rotations to get dynamic pitch
        dynamicpitch = RCfilter(dynamicpitch_unfilt, RC, fs)
        dynamicyaw_unfilt = integrate.cumtrapz(gz, dx=dt, axis=0) # time integrate z rotations to get dynamic yaw
        dynamicyaw = RCfilter(dynamicyaw_unfilt, RC, fs)
    else:
        dynamicroll = np.zeros(gx.shape)
        dynamicpitch = np.zeros(gy.shape)
        dynamicyaw = np.zeros(gz.shape)

    # Combine Orientation estimates using complimentary filter
    roll = Wd*dynamicroll + Ws*staticroll
    pitch = Wd*dynamicpitch + Ws*staticpitch
    yaw = Wd*dynamicyaw + Ws*staticyaw

    # Make rotation matrices at each time step 
    # references: Zippel 2018 (JPO) and Edson 1998
    # Transformation matrix T = A(yaw)A(pitch)A(roll)

    # Find number of samples from record length 
    samples = roll.shape[0]
    for n in np.arange(samples):
        # define roll(phi), pitch(theta) and yaw(psi) at time step
        phi = np.deg2rad(roll[n, 0])
        theta = np.deg2rad(pitch[n, 0])
        psi = np.deg2rad(yaw[n, 0])

        # Yaw Marix
        A_psi = np.array([[ np.cos(psi), np.sin(psi), 0], 
                          [-np.sin(psi), np.cos(psi), 0], 
                          [0,            0,           1]])
        
        # Pitch matrix
        A_theta = np.array([[np.cos(theta),  0, np.sin(theta)], 
                           [0,              1,             0], 
                           [-np.sin(theta), 0, np.cos(theta)]])

        # Roll Matrix
        A_phi = np.array([[1, 0,                      0], 
                          [0, np.cos(phi), -np.sin(phi)], 
                          [0, np.sin(phi),  np.cos(phi)]])

        # Multiply each transformation matrix together to get the overall transformation matrix
        T = np.dot(A_psi, np.dot(A_theta, A_phi))

        # Rotate each linear accelerations to the earth frame from the body frame
        a_vec = np.dot(T, np.array([[ax[n]], [ay[n]], [az[n]] ]).reshape(3,1)  )
        ax[n] = a_vec[0, 0]
        ay[n] = a_vec[1, 0]
        az[n] = a_vec[2, 0]

        # Rotate each magnetometer reading to the earth frame from the body frame
        m_vec = np.dot(T, np.array([[mx[n]], [my[n]], [mz[n]] ]).reshape(3,1)   )
        mx[n] = m_vec[0, 0]
        my[n] = m_vec[1, 0]
        mz[n] = m_vec[2, 0]

        # Create Angular rate matrix in earth frame and determine projected speeds
        # from Edson 1998 , rotate the "strapped-down" gyro measurements from body to earth frame

    # Define demean function
    def demean(x):
        x_demean = x - np.mean(x)
        return x_demean

    # Demean, filter and integrate linear accelerations to get linear velocities
    # Demean/ detrend
    ax = demean(ax)
    ay = demean(ay)
    az = demean(az)

    # Filter
    ax = RCfilter(ax, RC, fs)
    ay = RCfilter(ay, RC, fs)
    az = RCfilter(az, RC, fs)

    # Integrate 
    vx = integrate.cumtrapz(ax, dx=dt, axis=0)
    vy = integrate.cumtrapz(ay, dx=dt, axis=0)
    vz = integrate.cumtrapz(az, dx=dt, axis=0)

    # # Remove rotation-induced velocites from total velocity 
    # vx = vx - vxr
    # vy = vy - vyr
    # vz = vz - vzr

    # Determine geographic heading and correct horizontal velocities to East, North
    heading = np.rad2deg(np.arctan2(my + myo, mx + mxo))
    heading[heading < 0] = 360+heading[heading < 0]
    theta = -(heading - 90) # Cartesian CCW heading from geographic CW heading 

    # Compute east and north velocity components
    u = vx.copy() # x-direction, horizontal in earth frame but relative in azimuth
    print(u.shape)
    v = vy.copy() # y direction, horizontal in earth frame but relative in azimuth
    print(v.shape)
    vx = u * np.cos(np.deg2rad(theta)) - v * np.sin(np.deg2rad(theta)) # east component
    vy = u * np.sin(np.deg2rad(theta)) + v * np.cos(np.deg2rad(theta)) # north component

    # Demean, Filter and integrate Velocity to get displacement
    # Demean
    vx = demean(vx)
    vy = demean(vy)
    vz = demean(vz)

    # Filter 
    vx = RCfilter(vx, RC, fs)
    vy = RCfilter(vy, RC, fs)
    vz = RCfilter(vz, RC, fs)

    # Integrate
    x = integrate.cumtrapz(vx, dx=dt, axis=0)
    y = integrate.cumtrapz(vy, dx=dt, axis=0)
    z = integrate.cumtrapz(vz, dx=dt, axis=0)

    # Demean annd filter final signal
    x = RCfilter(x, RC, fs)
    y = RCfilter(y, RC, fs)
    z = RCfilter(z, RC, fs)

    # Remove first portion that has initial oscillations from filtering
    x[0:int(RC/(dt*10)), :] = 0
    y[0:int(RC/(dt*10)), :] = 0
    z[0:int(RC/(dt*10)), :] = 0

    # Return final values
    return x, y, z, roll, pitch, yaw, heading