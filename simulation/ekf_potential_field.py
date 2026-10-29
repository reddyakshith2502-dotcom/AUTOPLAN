import numpy as np

class ExtendedKalmanFilter2D:
    """
    Extended Kalman Filter (EKF) for Vehicle Localization & Uncertainty Propagation.
    State vector: x = [p_x, p_y, yaw, v]^T
    Covariance: P (4x4 matrix, with 2x2 spatial uncertainty submatrix P_xy)
    """
    def __init__(self, init_pos=(0, 0), init_yaw=0.0, init_v=0.0):
        self.state = np.array([float(init_pos[0]), float(init_pos[1]), float(init_yaw), float(init_v)])
        
        # Initial covariance matrix
        self.P = np.diag([0.25, 0.25, 0.05, 0.1])
        
        # Process noise matrix Q (odometry drift)
        self.Q = np.diag([0.08, 0.08, 0.02, 0.05])
        
        # Measurement noise matrix R (GNSS receiver noise)
        self.R_gps_normal = np.diag([0.15, 0.15])      # High-precision RTK
        self.R_gps_degraded = np.diag([6.5, 6.5])      # Weak GPS / Urban canyon

    def predict(self, dt=0.05, accel=0.0):
        """EKF Prediction Step: integrates wheel odometry and IMU"""
        px, py, yaw, v = self.state
        
        # Kinematic state transition
        px_new = px + v * np.cos(yaw) * dt
        py_new = py + v * np.sin(yaw) * dt
        v_new = max(0.0, v + accel * dt)
        
        self.state[0] = px_new
        self.state[1] = py_new
        self.state[3] = v_new
        
        # Jacobian of motion model F
        F = np.eye(4)
        F[0, 2] = -v * np.sin(yaw) * dt
        F[0, 3] = np.cos(yaw) * dt
        F[1, 2] = v * np.cos(yaw) * dt
        F[1, 3] = np.sin(yaw) * dt
        
        # Propagate covariance: P = F * P * F^T + Q
        self.P = F @ self.P @ F.T + self.Q

    def update_gps(self, measurement_xy, in_weak_gps_zone=False):
        """EKF Measurement Update: updates state and shrinks covariance if GPS is available"""
        if in_weak_gps_zone:
            # Degraded or zero GNSS updates -> covariance expands
            R = self.R_gps_degraded
        else:
            R = self.R_gps_normal
            
        H = np.zeros((2, 4))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        
        z = np.array([float(measurement_xy[0]), float(measurement_xy[1])])
        z_pred = H @ self.state
        y = z - z_pred  # Innovation / residual
        
        S = H @ self.P @ H.T + R  # Innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)  # Kalman Gain
        
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    def get_uncertainty_semi_axes(self):
        """
        Extracts 95% confidence (2-sigma) ellipse semi-major and semi-minor axes from P_xy.
        Returns: (major_axis_meters, minor_axis_meters, orientation_radians)
        """
        P_xy = self.P[:2, :2]
        eigenvals, eigenvecs = np.linalg.eigh(P_xy)
        
        # 2-sigma scale (chi-square 95% = 2.4477)
        scale = 2.4477
        major = float(np.sqrt(max(0.01, eigenvals[1])) * scale)
        minor = float(np.sqrt(max(0.01, eigenvals[0])) * scale)
        angle = float(np.arctan2(eigenvecs[1, 1], eigenvecs[0, 1]))
        
        return major, minor, angle


class ContinuousRiskPotentialField:
    """
    Artificial Potential Field (APF) for Spatio-Temporal Collision Risk Modeling.
    U_total(x, y) = U_repulsive_hazards(x, y) + U_repulsive_vehicles(x, y)
    """
    def __init__(self):
        self.hazard_sources = []  # list of (x, y, amplitude, sigma_x, sigma_y)

    def add_hazard(self, x, y, amplitude=10.0, radius=45.0):
        self.hazard_sources.append({
            'x': float(x),
            'y': float(y),
            'amplitude': float(amplitude),
            'sigma': float(radius)
        })

    def evaluate_risk(self, x, y):
        """Calculates scalar continuous repulsive potential U(x, y)"""
        total_risk = 0.0
        for src in self.hazard_sources:
            dx = x - src['x']
            dy = y - src['y']
            dist_sq = dx**2 + dy**2
            sigma_sq = src['sigma']**2
            potential = src['amplitude'] * np.exp(-dist_sq / (2.0 * sigma_sq))
            total_risk += potential
        return float(total_risk)
